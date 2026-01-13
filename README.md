Put the respective folder into the custom_nodes folder, and restart everything.  I've only tested the code from this repo on Ubuntu usign the ComfyUI install instructions provided on their github repo.


**Uses V3 Single GPU**
~16.5 GB = 1920x1088 800 frames
~18.5 GB = 1920x1088 900 frames

Mess around with ffn_chunks, 16 is what I used to get these results on a single 4090 gpu
