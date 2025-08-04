# 2025_Hypersonic_BL_ID
Hypersonic boundary layer normalization for transfer-learning cross-experimental applications

## BL Normalization Code
- BL_find: independent code to find boundary layer height and height of cone (if applicable).
	- input: raw .tiff files
	- output: boundary layer height, cone height
	- NOTES: change threshold until a reasonable boundary layer height is found
- total_normalization: normalizes boundary layer characteristics in image set
	- input: labeled image .txt file, boundary layer height found with BL_find.m, crop_rec used to crop cone out of image (use cone height to determine this found with BL_find.m)
	- output: .txt file with labeled normalized image data
- WL_find: independent code to find the wavelength of wave packets in labeled image sets
	- input: labeled image .txt file
	- output: average wavelength across labeled images
	



