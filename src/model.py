import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize
import numpy as np
import segmentation_models_pytorch as smp
import cv2

from skimage import io, measure
from shapely.geometry import LineString
from pathlib import Path

from dataset import create_tiles_with_labels, from_tiles_to_image, overlay_images
from urudendro.image import load_image, write_image
from urudendro.drawing import Drawing, Color



class UNET:
    def __init__(self, weights_path = "/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/unet_experiment/latest_model.pth" ,
                 tile_size=512, overlap=0.1, output_dir=None):
        self.model = self.load_model(weights_path)
        self.model.eval()
        self.tile_size = tile_size
        self.overlap = overlap
        self.output_dir = output_dir

    def load_model(self, weights_path):
        # Load your model here
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        model.load_state_dict(torch.load(weights_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        model = model.to(device)
        return model

    def forward(self, img, output_dir=None):
        if output_dir:
            write_image(f"{output_dir}/img.png", img)
        image, _ = create_tiles_with_labels(img, img, tile_size=self.tile_size, overlap=self.overlap)
        image = torch.from_numpy(image)

        with torch.no_grad():
            #split the image into tiles
            image = image.permute(0, 3, 1, 2).float() / 255.0  # Normalize to [0, 1]
            pred = self.model(image)
            pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
            pred = pred.squeeze().cpu().numpy()  # Convert to numpy array

            pred = from_tiles_to_image(pred, self.tile_size, img, self.overlap, output_dir=output_dir, img=img)
            write_image(f"{output_dir}/pred.png", (pred * 255).astype(np.uint8))

        #pred = (pred > 0.5).astype(np.uint8)  # Binarize the mask
        return pred

    def compute_connected_components(self, img):
        binary_image = img > 0  # Thresholding for binary conversion
        labels = measure.label(binary_image, connectivity=2)
        m_ch_e = []
        for region in measure.regionprops(labels):
            points = region.coords[:,[1,0]]#.tolist()
            #sort the points sequentially

            m_ch_e.extend(points + [[-1,-1]])
            m_ch_e.append(region.centroid)
        return np.array(m_ch_e)

    def compute_connected_components_by_contour(self, skeleton, output_dir, min_length_percentile=90):

        contours, hierarchy = cv2.findContours(skeleton.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        from urudendro.drawing import Drawing,Color
        from shapely.geometry import LineString
        img = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)
        color = Color()
        m_ch_e = []
        lenghts_contours = [c.shape[0] for c in contours]
        #sort contours by size descinding
        contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
        #get the 20 lower percentiles
        min_length = np.percentile(lenghts_contours, min_length_percentile)
        print(f"Min length: {min_length}. len(contours): {len(contours)}")

        #contours = [c for c in contours if c.shape[0] > min_length]
        print(f"Min length: {min_length}. len(contours): {len(contours)}")
        contour_img = np.zeros_like(skeleton) - 1
        for idx, c in enumerate(contours):
            c_shape = c.shape[0]
            if c_shape < 2:
                continue
            maximun_in_contour = contour_img[c[:,0][:,1], c[:,0][:,0]].max()
            if maximun_in_contour>-1:
                continue
            contour_img[c[:, 0][:, 1], c[:, 0][:, 0]] = idx
            poly = LineString(c[:,0][:,::-1])
            img = Drawing.curve(poly.coords, img, color.get_next_color())
            m_ch_e.extend(c[:,0].tolist() + [[-1,-1]])


        write_image(f"{output_dir}/contours.png", img)

        return np.array(m_ch_e)

    def gradient(self, img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
        return gx, gy

    def compute_normals(self, m_ch_e, h,w):
        n = len(m_ch_e)
        Gy = np.zeros((h,w))
        Gx = np.zeros((h,w))

        for idx in range(n):
            p0 = m_ch_e[idx]
            p_m1 = m_ch_e[idx - 1] if idx > 0 else m_ch_e[idx]
            p_m2 = m_ch_e[idx + 1] if idx < n - 1 else m_ch_e[idx]

            if p0[0] == -1:
                continue

            tg = np.array(p_m2) - np.array(p_m1)
            tgnorm = tg / np.linalg.norm(tg)
            normal = np.array([-tgnorm[1], tgnorm[0]])

            i = int(p0[0])
            j = int(p0[1])
            Gy[j,i] = normal[0]
            Gx[j,i] = normal[1]


        return Gx, Gy




def rotate_image(image, center, angle=90):
    # Get image dimensions and calculate rotation matrix
    if angle == 0:
        return image
    (h, w) = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform rotation
    rotated_image = cv2.warpAffine(image.copy(), rotation_matrix, (w, h))
    return rotated_image

def deep_learning_edge_detector(img,
                                weights_path= "/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/unet_experiment/latest_model.pth",
                                output_dir=None, cy=None, cx=None):
    model = UNET(weights_path)

    angle_range = [0,90,180,270]
    #angle_range = [0]
    pred_dict = {}
    for angle in angle_range:
        output_dir_angle = Path(output_dir) / f"{angle}"
        output_dir_angle.mkdir(parents=True, exist_ok=True)
        output_dir_angle = None
        rot_image = rotate_image(img, (cx, cy), angle=angle)
        pred = model.forward(rot_image, output_dir=output_dir_angle)
        pred = rotate_image(pred, (cx, cy), angle=-angle)
        pred_dict[angle] = pred
    # Combine the predictions
    pred = np.zeros_like(pred_dict[angle])
    for angle in angle_range:
        pred += pred_dict[angle]

    #clip the values to 0
    # scale the values to 0-255
    if output_dir:
        pred_scaled = (pred * 255).astype(np.uint8)
        #overlay the prediction on the image
        overlay = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        overlay[:,:,0] = pred_scaled
        overlay = overlay_images(img.astype(np.uint8), overlay, alpha=0.5, beta=0.5, gamma=0)
        write_image(f"{output_dir}/overlay.png", overlay)

    #binarize the mask
    th = len(angle_range) / 3
    pred = (pred >= th).astype(np.uint8)  # Binarize the mask
    skeleton = skeletonize(pred)
    skeleton = np.where(skeleton, 255, 0)  # Skeletonize the mask

    # write the image to disk
    if output_dir:
        write_image(f"{output_dir}/skel.png", skeleton)
    #m_ch_e = model.compute_connected_components(skeleton)
    m_ch_e = model.compute_connected_components_by_contour(skeleton, output_dir)
    gx, gy = model.compute_normals(m_ch_e, img.shape[0], img.shape[1])
    if output_dir:
        #write_image("labels.png", labels)
        debug_image = img.copy()

        amp = 10
        for idx in range(len(m_ch_e)):
            p0 = m_ch_e[idx]
            i = int(p0[0])
            j = int(p0[1])

            p1 = p0 + amp*np.array([gy[j,i], gx[j,i]])
            p1 = p1.astype(int)
            if p0[0] < 1 or p0[1] < 1 or p1[0] < 1 or p1[1] < 1:
                continue
            points = np.stack([p0,p1])[:,::-1]
            line = LineString(points)
            debug_image = Drawing.radii(line.coords, debug_image, color=Color.blue, thickness=1)

        for p in m_ch_e:
            if p[0] == -1:
                continue
            debug_image = Drawing.circle(debug_image, p, thickness=-1, color=Color.black, radius=1)

        write_image(f"{output_dir}/normals.png", debug_image)

    return m_ch_e, gy, gx
def test_forward(debug=False):
    weights_path = "/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/unet_experiment/latest_model.pth"
    image_path = "/data/maestria/resultados/deep_cstrd/pinus_v1/val/images/segmented/F02d.png"
    img = load_image(image_path)
    #convert to torch tensor
    m_ch_e = deep_learning_edge_detector(img, weights_path)

    return m_ch_e



if __name__ == "__main__":
    test_forward()