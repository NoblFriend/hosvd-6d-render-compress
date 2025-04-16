import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from scipy.linalg import svd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

# ------------------------
# Параметры анимации и модели
frames = 20
interval = 200
radius = 1.0
h_top = 2
size=64
dtheta = 2 * np.pi / frames
azimuth_values = [0, 45, 90]
elevation_values = [-60, 0, 30, 80]
face_colors = ['lightblue', 'lightgreen', 'lightcoral', 'wheat']

# ------------------------
def get_vertices(theta):
    angles = np.array([theta, theta + 2*np.pi/3, theta + 4*np.pi/3])
    base_points = np.column_stack((radius * np.cos(angles),
                                   radius * np.sin(angles),
                                   np.zeros(3)))
    top_point = np.array([0, 0, h_top])
    return base_points, top_point

def get_faces(base_points, top_point):
    faces = [[base_points[0], base_points[1], base_points[2]]]
    for i in range(3):
        j = (i + 1) % 3
        faces.append([base_points[i], base_points[j], top_point])
    return faces

def update_plot(ax, theta, show_axes=True, azim=None, elev=None):
    ax.cla()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-0.5, 2.5)
    ax.set_box_aspect([1, 1, 1])
    base_points, top_point = get_vertices(theta)
    faces = get_faces(base_points, top_point)
    pts = np.vstack((base_points, top_point))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='k', s=50)
    poly3d = Poly3DCollection(faces, linewidths=1, edgecolors='k')
    poly3d.set_facecolor(face_colors)
    ax.add_collection3d(poly3d)
    if not show_axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.grid(False); ax.set_axis_off()
    if azim is not None and elev is not None:
        ax.view_init(elev=elev, azim=azim)

def render_view_frame(view, theta, show_axes=False):
    """
    Отрисовывает пирамиду в заданном виде (view=(азимут, elevation)) и возвращает изображение 256x256 (RGB).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    azim, elev = view
    ax.view_init(elev=elev, azim=azim)
    
    update_plot(ax, theta, show_axes=show_axes, azim=azim, elev=elev)
    fig.canvas.draw()
    
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    reported_pixels = width * height
    num_pixels = buf.size // 4
    scale = int(np.sqrt(num_pixels / reported_pixels))
    new_height = height * scale
    new_width  = width * scale

    try:
        img = buf.reshape((new_height, new_width, 4))
    except Exception as e:
        plt.close(fig)
        raise ValueError(f"Ошибка изменения формы буфера: {e}")
    
    plt.close(fig)
    img = img[:, :, :3]  # отбрасываем альфа-канал
    im = Image.fromarray(img).resize((size,size))
    return np.array(im)

def save_tensor():
    num_az = len(azimuth_values)
    num_el = len(elevation_values)
    tensor = np.zeros((size, size, 3, num_az, num_el, frames), dtype=np.uint8)
    for i, az in enumerate(azimuth_values):
        for j, elev in enumerate(elevation_values):
            for t in range(frames):
                theta = t * dtheta
                img = render_view_frame((az, elev), theta, show_axes=False)
                tensor[:, :, :, i, j, t] = img
    np.save("tensor.npy", tensor)

def animate_tensor(tensor, filename="tensor_animation.gif", save_frames=False):
    H, W, C, n_rows, n_cols, num_frames = tensor.shape
    frame_dir = filename.split(".")[0]  # example: original_tensor → folder name

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axs = np.atleast_2d(axs)

    if save_frames:
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        for frame in range(num_frames):
            for i in range(n_rows):
                for j in range(n_cols):
                    axs[i, j].clear()
                    axs[i, j].imshow(tensor[:, :, :, i, j, frame])
                    axs[i, j].axis('off')
            plt.suptitle(f"Frame {frame}")
            plt.savefig(f"{frame_dir}/frame_{frame:03d}.png", bbox_inches='tight')
        print(f"PNG-кадры сохранены в папку: {frame_dir}")

    # Потом делаем gif отдельно (уже без сохранения PNG)
    def animate(frame):
        for i in range(n_rows):
            for j in range(n_cols):
                axs[i, j].clear()
                axs[i, j].imshow(tensor[:, :, :, i, j, frame])
                axs[i, j].axis('off')
        plt.suptitle(f"Frame {frame}")

    anim = FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=False)
    anim.save(filename, writer=PillowWriter(fps=1000 // interval))
    plt.close(fig)
    print(f"GIF сохранен: {filename}")



def unfold(tensor, mode):
    sz = range(len(tensor.shape))
    new_tensor = np.moveaxis(tensor, sz, np.roll(sz, mode))
    return np.reshape(new_tensor, (tensor.shape[mode], -1))

def hosvd_decompose(tensor):
    U_matrices = []
    core_tensor = tensor.copy()
    for mode in range(len(tensor.shape)):
        U, _, _ = svd(unfold(tensor, mode), full_matrices=False)
        U_matrices.append(U)
        core_tensor = np.moveaxis(np.tensordot(core_tensor, U.T, axes=[mode, 1]), -1, mode)
    return core_tensor, U_matrices


def hosvd_reconstruct(core_tensor, U_matrices, ranks):
    tensor = core_tensor.copy()
    for mode in range(len(ranks)):
        U = U_matrices[mode]
        U_truncated = np.zeros_like(U)
        min_dim = min(U.shape[1], ranks[mode])
        U_truncated[:, :min_dim] = U[:, :min_dim]
        tensor = np.moveaxis(np.tensordot(tensor, U_truncated, axes=[mode, 1]), -1, mode)
    return tensor


def hosvd_approx(tensor, ranks):
    U_matrices = []
    core_tensor = tensor
    for mode in range(len(tensor.shape)):
        U, _, _ = svd(unfold(tensor, mode), full_matrices=False)
        U_full = np.zeros_like(U)
        min_dim = min(U.shape[1], ranks[mode])
        U_full[:, :min_dim] = U[:, :min_dim]
        U_matrices.append(U_full)
        core_tensor = np.moveaxis(np.tensordot(core_tensor, U_full.T, axes=[mode, 1]), -1, mode)
    tensor = core_tensor
    for mode, U in enumerate(U_matrices):
        tensor = np.moveaxis(np.tensordot(tensor, U, axes=[mode, 1]), -1, mode)
    return tensor

def restore_color_range(tensor):
    tensor = np.clip(tensor, 0, 255)
    return tensor.astype(np.uint8)


def restore_color_range(tensor):
    tensor = np.clip(tensor, 0, 255)
    return tensor.astype(np.uint8)

def experiment():
    tensor = np.load("tensor.npy")
    original = tensor.astype(np.float64)
    shape = tensor.shape
    core, U_matrices = hosvd_decompose(original)
    results = {}

    def compute_metrics(approx, tag):
        psnr_vals = []
        ssim_vals = []
        for i in range(shape[3]):
            for j in range(shape[4]):
                for t in range(shape[5]):
                    img_true = tensor[:, :, :, i, j, t]
                    img_pred = approx[:, :, :, i, j, t]
                    psnr_vals.append(peak_signal_noise_ratio(img_true, img_pred))
                    ssim_vals.append(structural_similarity(img_true, img_pred, channel_axis=2))
        psnr_mean, psnr_std = np.mean(psnr_vals), np.std(psnr_vals)
        ssim_mean, ssim_std = np.mean(ssim_vals), np.std(ssim_vals)
        results[tag] = (psnr_mean, psnr_std, ssim_mean, ssim_std)

    for spatial_rank in tqdm([64, 48, 32, 24, 16]):
        ranks = [spatial_rank, spatial_rank, 3, 3, 4, 20]
        approx = hosvd_reconstruct(core, U_matrices, ranks)
        approx = restore_color_range(approx)
        compute_metrics(approx, f"spatial_{spatial_rank}")

    for az_rank in tqdm([3, 2, 1]):
        ranks = [64, 64, 3, az_rank, 4, 20]
        approx = hosvd_reconstruct(core, U_matrices, ranks)
        approx = restore_color_range(approx)
        compute_metrics(approx, f"azimuth_{az_rank}")

    for el_rank in tqdm([4, 3, 2, 1]):
        ranks = [64, 64, 3, 3, el_rank, 20]
        approx = hosvd_reconstruct(core, U_matrices, ranks)
        approx = restore_color_range(approx)
        compute_metrics(approx, f"elevation_{el_rank}")

    for time_rank in tqdm([20, 16, 12, 8, 4]):
        ranks = [64, 64, 3, 3, 4, time_rank]
        approx = hosvd_reconstruct(core, U_matrices, ranks)
        approx = restore_color_range(approx)
        compute_metrics(approx, f"time_{time_rank}")

    def plot_all_metrics(results, full_ranks):
        prefixes = ["spatial", "azimuth", "elevation", "time"]

        fig_psnr, ax_psnr = plt.subplots(figsize=(8, 5))
        fig_ssim, ax_ssim = plt.subplots(figsize=(8, 5))

        for prefix in prefixes:
            keys = sorted(
                [k for k in results if k.startswith(prefix)],
                key=lambda x: int(x.split('_')[-1])
            )
            if not keys:
                continue

            full_rank = full_ranks[prefix]
            xs = [int(k.split('_')[-1]) / full_rank for k in keys]
            psnr_vals = [results[k][0] for k in keys]
            psnr_errs = [results[k][1] for k in keys]
            ssim_vals = [results[k][2] for k in keys]
            ssim_errs = [results[k][3] for k in keys]

            ax_psnr.errorbar(
                xs, psnr_vals, yerr=psnr_errs,
                fmt='o', capsize=4, label=prefix,
                linestyle='--', linewidth=1, markersize=6
            )
            ax_ssim.errorbar(
                xs, ssim_vals, yerr=ssim_errs,
                fmt='o', capsize=4, label=prefix,
                linestyle='--', linewidth=1, markersize=6
            )

        ax_psnr.set_title("PSNR vs Доля сохранённого ранга")
        ax_psnr.set_xlabel("Доля сохранённого ранга")
        ax_psnr.set_ylabel("PSNR (дБ)")
        ax_psnr.legend()
        ax_psnr.grid(True)
        fig_psnr.tight_layout()
        fig_psnr.savefig("psnr_all.png")
        plt.close(fig_psnr)

        ax_ssim.set_title("SSIM vs Доля сохранённого ранга")
        ax_ssim.set_xlabel("Доля сохранённого ранга")
        ax_ssim.set_ylabel("SSIM")
        ax_ssim.legend()
        ax_ssim.grid(True)
        fig_ssim.tight_layout()
        fig_ssim.savefig("ssim_all.png")
        plt.close(fig_ssim)


    full_ranks = {
        "spatial": 64,
        "azimuth": 3,
        "elevation": 4,
        "time": 20
    }

    plot_all_metrics(results, full_ranks)

def save_tensor_frames(tensor, frame_dir="frames", prefix="frame"):
    H, W, C, n_rows, n_cols, num_frames = tensor.shape

    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axs = np.atleast_2d(axs)

    for frame in range(num_frames):
        for i in range(n_rows):
            for j in range(n_cols):
                axs[i, j].clear()
                axs[i, j].imshow(tensor[:, :, :, i, j, frame])
                axs[i, j].axis('off')
        plt.suptitle(f"Frame {frame}", fontsize=16)
        filepath = os.path.join(frame_dir, f"{prefix}_{frame:03d}.png")
        plt.savefig(filepath, bbox_inches='tight')
        print(f"saved: {filepath}")

    plt.close(fig)
    
def main():
    if not os.path.exists("tensor.npy"):
        save_tensor()
    tensor = np.load("tensor.npy")


    ranks = [256, 256, 3, 2, 2, 50]
    tensor_approx = hosvd_approx(tensor, ranks)
    tensor_approx = restore_color_range(tensor_approx)
    np.save("reduced_tensor.npy", tensor_approx.astype(np.uint8))

    save_tensor_frames(tensor_approx, frame_dir="reduced_frames", prefix="reduced")
    animate_tensor(tensor_approx, filename="reduced_tensor.gif")
    save_tensor_frames(tensor, frame_dir="original_frames", prefix="orig")
    animate_tensor(tensor, filename="original_tensor.gif")


if __name__ == '__main__':
    main()
    experiment()