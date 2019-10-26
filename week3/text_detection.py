import os
from glob import glob
import cv2
from textbox_removal import TextBoxRemoval
from noise import Denoise

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def detect_text(text_img_path):
    return pytesseract.image_to_string(Image.open(text_img_path))


def get_text_single_painting_img(img,img_path):
    print("Detecting textbox of image",img_path)
    mask, textbox = TextBoxRemoval(img)
    bbox = [textbox[0][1], textbox[0][0], textbox[1][1], textbox[1][0]]
    cv2.imwrite(img_path.replace(".jpg", "_textboxmask.png"), cv2.resize(mask,(1000,1000)))
    text_img = cv2.bitwise_and(img, img, mask=~mask)
    cropped_text = img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    text_img_path = img_path.replace(".jpg", "_text.png")
    cv2.imwrite(text_img_path, cropped_text)
    text = detect_text(text_img_path)

    # text = detect_text(img_path.replace(".jpg","_denoised.png"))
    return text


def main(img_folder):
    denoiser = Denoise(img_folder)
    print("Denoising images...")
    # denoised_imgs = denoiser.bilateral(win_size=None,sigma_spatial=1,bins=5000)
    # denoised_imgs = denoiser.nl_means(patch_size=7,patch_distance=11,cut_off=0.1,fast_mode=True,sigma=0.0)
    # denoised_imgs = denoiser.fast_nlmean(h=10,hC=10,template_size=7,win_size=21)
    denoised_imgs = denoiser.tv_bregman(weight=0.01,max_iter=1000,eps=0.001,isotropic=True)
    # denoised_imgs = denoiser.tv_chambolle(weight=10,eps=0.001,max_iter=1000)
    # denoised_imgs = denoiser.wavelet(sigma=None,wavelet='db1',mode='soft',wav_lev=None,method='BayesShrink')

    # denoised_imgs = [cv2.medianBlur(cv2.imread(item),5) for item in sorted(glob(os.path.join(img_folder, "*.jpg")))]
    # denoised_imgs = [cv2.GaussianBlur(item,(3,3),0) for item in denoised_imgs]
    print("Done.")
    for img_path,img in zip(sorted(glob(os.path.join(img_folder, "*.jpg"))),denoised_imgs):
        cv2.imwrite(img_path.replace(".jpg","_denoised.png"),img)
        print(img_path,"Text detected:",get_text_single_painting_img(img,img_path))


if __name__ == '__main__':
    imgs_folder = r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week3\tests_folder"
    main(imgs_folder)
