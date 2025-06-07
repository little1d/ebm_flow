import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import math
from pathlib import Path

# --- 1. 辅助模块和U-Net构建 ---


class SinusoidalPositionEmbeddings(nn.Module):
    """
    将时间步 t 转换为正弦位置嵌入。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ConditionalUNet(nn.Module):
    """
    带类别条件的U-Net模型。
    """

    def __init__(
        self,
        dim=64,
        num_classes=10,
        channels=1,
        dim_mults=(1, 2, 4),
    ):
        super().__init__()
        self.channels = channels
        init_dim = dim
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 时间和类别嵌入
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_emb = nn.Embedding(num_classes, time_dim)

        # 下采样路径
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        Block(dim_in, dim_out),
                        Block(dim_out, dim_out),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 中间层
        mid_dim = dims[-1]
        self.mid_block1 = Block(mid_dim, mid_dim)
        self.mid_block2 = Block(mid_dim, mid_dim)

        # 上采样路径
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        Block(dim_out * 2, dim_in),
                        Block(dim_in, dim_in),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 最终层
        self.final_conv = nn.Sequential(
            Block(dim, dim), nn.Conv2d(dim, self.channels, 1)
        )

    def forward(self, x, time, y):
        x = self.init_conv(x)
        t_emb = self.time_mlp(time)
        y_emb = self.class_emb(y)
        emb = t_emb + y_emb

        h = []

        for block1, block2, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_block2(x)

        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)
            x = block2(x)
            x = upsample(x)

        return self.final_conv(x)


# --- 2. 扩散过程辅助函数 ---


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class Diffusion:
    """
    管理扩散过程的辅助类。
    """

    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = linear_beta_schedule(timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

    # 前向过程: q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    # 反向过程: p(x_{t-1} | x_t)
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, y):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, y) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1, y=None):
        shape = (batch_size, channels, image_size, image_size)
        img = torch.randn(shape, device=self.device)
        imgs = []

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="Sampling",
            total=self.timesteps,
        ):
            t = torch.full(
                (batch_size,), i, device=self.device, dtype=torch.long
            )
            img = self.p_sample(model, img, t, i, y)
            imgs.append(img.cpu())
        return imgs


# --- 3. 训练和采样 ---
def train(epochs=20):
    # --- 配置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    learning_rate = 1e-3
    image_size = 28
    channels = 1
    timesteps = 1000

    # --- 结果保存路径 ---
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    model_path = results_folder / "conditional_ddpm_mnist.pth"

    # --- 数据加载 ---
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),  # 归一化到 [-1, 1]
        ]
    )
    dataset = datasets.MNIST(
        root='/Users/little1d/Desktop/Playground/ebm_flow/data',
        train=True,
        download=True,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- 初始化 ---
    model = ConditionalUNet(dim=64, channels=channels, num_classes=10).to(
        device
    )
    diffusion = Diffusion(timesteps=timesteps, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 训练循环 ---
    for epoch in range(epochs):
        for step, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]

            # 1. 采样时间步 t
            t = torch.randint(
                0, timesteps, (batch_size,), device=device
            ).long()

            # 2. 加噪
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(x_start=images, t=t, noise=noise)

            # 3. 预测噪声
            predicted_noise = model(x_t, t, labels)

            # 4. 计算损失
            loss = F.mse_loss(noise, predicted_noise)

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch} | Step {step:03d} | Loss: {loss.item():.4f}"
                )

        # 每个epoch结束后保存一张采样图
        if epoch % 5 == 0 or epoch == epochs - 1:
            generate_and_save_samples(
                model, diffusion, epoch, image_size, device
            )

    # --- 保存模型 ---
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def generate_and_save_samples(model, diffusion, epoch, image_size, device):
    model.eval()
    # 为每个类别生成一张图
    labels_to_generate = torch.arange(10).to(device)
    num_images = len(labels_to_generate)

    with torch.no_grad():
        generated_images_steps = diffusion.sample(
            model,
            image_size,
            batch_size=num_images,
            channels=1,
            y=labels_to_generate,
        )
        final_images = generated_images_steps[-1]

    # 反归一化到 [0, 1]
    final_images = (final_images + 1) * 0.5
    save_image(
        final_images, f"./results/mnist_generated_epoch_{epoch}.png", nrow=10
    )
    print(f"Generated samples saved for epoch {epoch}")
    model.train()


if __name__ == "__main__":
    train(epochs=20)

    # 训练结束后，生成最终的图像网格
    print("\nTraining finished. Generating final image grid...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 28
    model = ConditionalUNet(dim=64, channels=1, num_classes=10).to(device)
    model.load_state_dict(torch.load("./results/conditional_ddpm_mnist.pth"))
    diffusion = Diffusion(timesteps=1000, device=device)

    generate_and_save_samples(model, diffusion, "final", image_size, device)
