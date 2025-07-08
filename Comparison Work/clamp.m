function out_img = clamp(in_img)

out_img = (in_img - min(min(in_img)))/(max(max(in_img)) -min(min(in_img)));