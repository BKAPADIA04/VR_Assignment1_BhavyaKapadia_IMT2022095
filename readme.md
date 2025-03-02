
# VR_Assignment1_BhavyaKapadia_IMT2022095

### git clone https://github.com/BKAPADIA04/Assignment-1.git

`cd Assignment-1`

### Part 1 : Detection, Segmentation And Coins Count

`cd Part1`

Requirements - `python`,`numpy` and `opencv` installed

```bash
pip install numpy opencv-python opencv-python-headless
```

### To run the code : 

```bash 
python IMT2022095_part1.py
```

#### Input Images are present in ./input_images (eg : 1.jpg,2.jpg,...,6.jpg)
#### Output Images are stored in ./output_images

##### The output images have images stored as (eg for 1.jpg):

##### The detected coins are saved as : detected_i.jpg (eg : detected_1.jpg)

##### The segmented coins are saved as : segmented_i.jpg (eg : segmented_1.jpg)

##### Each segmented coin is saved as : coin_i_j.jpg where j ranges from 1 to total coins detected. (eg : coin_1_1.jpg,coin_1_2.jpg,...,coin_1_6.jpg)

### Part 2 : Stitched Panorama From Multiple Images

`cd Part2`

Requirements - `python`,`numpy` and `opencv` installed

```bash
pip install numpy opencv-python opencv-python-headless
```

### To run the code : 
#### Input images are stored in input folder : input (eg:0.png,1.png,..,3.png)
#### Make a folder for storing the output images : output (Here I have already made it)
```bash 
python IMT2022095_part2.py
```

##### The output images are saved as :

##### Each image stitching keypoint matching is stored as : match_step_i.jpg (i ranges from 1 to total images - 1)

##### Final Panorama is saved as : stitched_panorama.jpg


