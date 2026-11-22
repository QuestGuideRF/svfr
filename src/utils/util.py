import os
from pathlib import Path
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
import imageio
def seed_everything (seed ):
    import random
    import numpy as np
    torch .manual_seed (seed )
    torch .cuda .manual_seed_all (seed )
    np .random .seed (seed %(2 **32 ))
    random .seed (seed )
def save_videos_from_pil (pil_images ,path ,fps =8 ):
    save_fmt =Path (path ).suffix
    os .makedirs (os .path .dirname (path ),exist_ok =True )
    if save_fmt ==".mp4":
        with imageio .get_writer (path ,fps =fps )as writer :
            for img in pil_images :
                img_array =np .array (img )
                writer .append_data (img_array )
    elif save_fmt ==".gif":
        pil_images [0 ].save (
        fp =path ,
        format ="GIF",
        append_images =pil_images [1 :],
        save_all =True ,
        duration =(1 /fps *1000 ),
        loop =0 ,
        optimize =False ,
        lossless =True
        )
    else :
        raise ValueError ("Неподдерживаемый тип файла. Используйте .mp4 или .gif.")
def save_videos_grid (videos :torch .Tensor ,path :str ,rescale =False ,n_rows =6 ,fps =8 ):
    videos =rearrange (videos ,"б с т ч ш -> т б ч ч ш")
    height ,width =videos .shape [-2 :]
    outputs =[]
    for i ,x in enumerate (videos ):
        x =torchvision .utils .make_grid (x ,nrow =n_rows )
        x =x .transpose (0 ,1 ).transpose (1 ,2 ).squeeze (-1 )
        if rescale :
            x =(x +1.0 )/2.0
        x =(x *255 ).numpy ().astype (np .uint8 )
        x =Image .fromarray (x )
        outputs .append (x )
    os .makedirs (os .path .dirname (path ),exist_ok =True )
    save_videos_from_pil (outputs ,path ,fps )