import argparse
import matplotlib.pyplot as plt
import os
import torch

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                    help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()


colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if opt.use_gpu:
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()


img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
if opt.use_gpu:
    tens_l_rs = tens_l_rs.cuda()


img_bw = postprocess_tens(tens_l_orig,
                          torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig,
                                  colorizer_eccv16(tens_l_rs).cpu())
out_img_siggraph17 = postprocess_tens(tens_l_orig,
                                      colorizer_siggraph17(tens_l_rs).cpu())


output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

eccv16_path = os.path.join(output_folder, f"{opt.save_prefix}_eccv16.png")
siggraph17_path = os.path.join(output_folder, f"{opt.save_prefix}_siggraph17.png")

plt.imsave(eccv16_path, out_img_eccv16)
plt.imsave(siggraph17_path, out_img_siggraph17)

print(f"Saved ECCV16 result at {eccv16_path}")
print(f"Saved SIGGRAPH17 result at {siggraph17_path}")

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_bw)
plt.title('Input (B&W)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')

plt.show()
