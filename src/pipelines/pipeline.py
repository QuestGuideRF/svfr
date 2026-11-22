import inspect
from dataclasses import dataclass
from typing import Callable ,Dict ,List ,Optional ,Union
from einops import rearrange
import numpy as np
import PIL .Image
import torch
from transformers import CLIPImageProcessor ,CLIPVisionModelWithProjection
from diffusers .image_processor import VaeImageProcessor
from diffusers .utils import BaseOutput ,logging
from diffusers .utils .torch_utils import randn_tensor ,is_compiled_module
from diffusers .pipelines .pipeline_utils import DiffusionPipeline
from diffusers import (
AutoencoderKLTemporalDecoder ,
EulerDiscreteScheduler ,
)
from src .models .svfr_adapter .unet_3d_svd_condition_ip import UNet3DConditionSVDModel
logger =logging .get_logger (__name__ )
def _append_dims (x ,target_dims ):
    dims_to_append =target_dims -x .ndim
    if dims_to_append <0 :
        raise ValueError (f"input has {x .ndim } dims but target_dims is {target_dims }, which is less")
    return x [(...,)+(None ,)*dims_to_append ]
def tensor2vid (video :torch .Tensor ,processor :VaeImageProcessor ,output_type :str ="np"):
    batch_size ,channels ,num_frames ,height ,width =video .shape
    outputs =[]
    for batch_idx in range (batch_size ):
        batch_vid =video [batch_idx ].permute (1 ,0 ,2 ,3 )
        batch_output =processor .postprocess (batch_vid ,output_type )
        outputs .append (batch_output )
    if output_type =="np":
        outputs =np .stack (outputs )
    elif output_type =="pt":
        outputs =torch .stack (outputs )
    elif not output_type =="pil":
        raise ValueError (f"{output_type } does not exist. Please choose one of ['np', 'pt', 'pil']")
    return outputs
@dataclass
class LQ2VideoSVDPipelineOutput (BaseOutput ):
    r
    frames :Union [List [PIL .Image .Image ],np .ndarray ]
    latents :Union [torch .Tensor ,np .ndarray ]
class LQ2VideoLongSVDPipeline (DiffusionPipeline ):
    r
    model_cpu_offload_seq ="image_encoder->unet->vae"
    _callback_tensor_inputs =["latents"]
    def __init__ (
    self ,
    vae :AutoencoderKLTemporalDecoder ,
    image_encoder :CLIPVisionModelWithProjection ,
    unet :UNet3DConditionSVDModel ,
    scheduler :EulerDiscreteScheduler ,
    feature_extractor :CLIPImageProcessor ,
    ):
        super ().__init__ ()
        self .register_modules (
        vae =vae ,
        image_encoder =image_encoder ,
        unet =unet ,
        scheduler =scheduler ,
        feature_extractor =feature_extractor ,
        )
        self .vae_scale_factor =2 **(len (self .vae .config .block_out_channels )-1 )
        self .image_processor =VaeImageProcessor (
        vae_scale_factor =self .vae_scale_factor ,
        do_convert_rgb =True )
    def _clip_encode_image (self ,image ,num_frames ,device ,num_videos_per_prompt ,do_classifier_free_guidance ):
        dtype =next (self .image_encoder .parameters ()).dtype
        if not isinstance (image ,torch .Tensor ):
            image =self .image_processor .pil_to_numpy (image )
            image =self .image_processor .numpy_to_pt (image )
            image =image *2.0 -1.0
            image =_resize_with_antialiasing (image ,(224 ,224 ))
            image =(image +1.0 )/2.0
            image =self .feature_extractor (
            images =image ,
            do_normalize =True ,
            do_center_crop =False ,
            do_resize =False ,
            do_rescale =False ,
            return_tensors ="pt",
            ).pixel_values
        image =image .to (device =device ,dtype =dtype ,non_blocking =True ,).unsqueeze (0 )
        image_embeddings =self .image_encoder (image ).image_embeds
        image_embeddings =image_embeddings .unsqueeze (1 )
        bs_embed ,seq_len ,_ =image_embeddings .shape
        image_embeddings =image_embeddings .repeat (1 ,num_videos_per_prompt ,1 )
        image_embeddings =image_embeddings .view (bs_embed *num_videos_per_prompt ,seq_len ,-1 )
        if do_classifier_free_guidance :
            negative_image_embeddings =torch .zeros_like (image_embeddings )
            image_embeddings =torch .cat ([negative_image_embeddings ,image_embeddings ])
        return image_embeddings
    def _encode_vae_image (
    self ,
    image :torch .Tensor ,
    device ,
    num_videos_per_prompt ,
    do_classifier_free_guidance ,
    ):
        image =image .to (device =device )
        image_latents =self .vae .encode (image ).latent_dist .mode ()
        image_latents =image_latents .unsqueeze (0 )
        if do_classifier_free_guidance :
            negative_image_latents =torch .zeros_like (image_latents )
            image_latents =torch .cat ([image_latents ,image_latents ])
        image_latents =image_latents .repeat (num_videos_per_prompt ,1 ,1 ,1 ,1 )
        return image_latents
    def _get_add_time_ids (
    self ,
    task_id_input ,
    dtype ,
    batch_size ,
    num_videos_per_prompt ,
    do_classifier_free_guidance ,
    ):
        passed_add_embed_dim =self .unet .config .addition_time_embed_dim *len (task_id_input )
        expected_add_embed_dim =self .unet .add_embedding .linear_1 .in_features
        if expected_add_embed_dim !=passed_add_embed_dim :
            raise ValueError (
            f"Model expects an added time embedding vector of length {expected_add_embed_dim }, but a vector of {passed_add_embed_dim } was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
        add_time_ids =task_id_input .to (dtype )
        add_time_ids =add_time_ids .repeat (batch_size *num_videos_per_prompt ,1 )
        if do_classifier_free_guidance :
            add_time_ids =torch .cat ([add_time_ids ,add_time_ids ])
        return add_time_ids
    def decode_latents (self ,latents ,num_frames ,decode_chunk_size =14 ):
        latents =latents .flatten (0 ,1 )
        latents =1 /self .vae .config .scaling_factor *latents
        forward_vae_fn =self .vae ._orig_mod .forward if is_compiled_module (self .vae )else self .vae .forward
        accepts_num_frames ="num_frames"in set (inspect .signature (forward_vae_fn ).parameters .keys ())
        frames =[]
        for i in range (0 ,latents .shape [0 ],decode_chunk_size ):
            num_frames_in =latents [i :i +decode_chunk_size ].shape [0 ]
            decode_kwargs ={}
            if accepts_num_frames :
                decode_kwargs ["num_frames"]=num_frames_in
            frame =self .vae .decode (latents [i :i +decode_chunk_size ],**decode_kwargs ).sample
            frames .append (frame )
        frames =torch .cat (frames ,dim =0 )
        frames =frames .reshape (-1 ,num_frames ,*frames .shape [1 :]).permute (0 ,2 ,1 ,3 ,4 )
        frames =frames .float ()
        return frames
    def check_inputs (self ,image ,height ,width ):
        if (
        not isinstance (image ,torch .Tensor )
        and not isinstance (image ,PIL .Image .Image )
        and not isinstance (image ,list )
        ):
            raise ValueError (
            "`image` должен иметь тип `torch.FloatTensor` или `PIL.Image.Image` или `List[PIL.Image.Image]`, но"
            f" {type (image )}"
            )
        if height %8 !=0 or width %8 !=0 :
            raise ValueError (f"`height` and `width` have to be divisible by 8 but are {height } and {width }.")
    def prepare_latents (
    self ,
    batch_size ,
    num_frames ,
    num_channels_latents ,
    height ,
    width ,
    dtype ,
    device ,
    generator ,
    latents =None ,
    ref_image_latents =None ,
    timestep =None
    ):
        from src .utils .noise_util import random_noise
        shape =(
        batch_size ,
        num_frames ,
        num_channels_latents //3 ,
        height //self .vae_scale_factor ,
        width //self .vae_scale_factor ,
        )
        if isinstance (generator ,list )and len (generator )!=batch_size :
            raise ValueError (
            f"You have passed a list of generators of length {len (generator )}, but requested an effective batch"
            f" size of {batch_size }. Make sure the batch size matches the length of the generators."
            )
        if latents is None :
            noise =random_noise (shape =shape ,generator =generator ,device =device ,dtype =dtype )
        else :
            noise =latents .to (device )
        if timestep is not None :
            init_latents =ref_image_latents .unsqueeze (0 )
            latents =self .scheduler .add_noise (init_latents ,noise ,timestep )
        else :
            latents =noise *self .scheduler .init_noise_sigma
        return latents
    def get_timesteps (self ,num_inference_steps ,strength ,device ):
        init_timestep =min (int (num_inference_steps *strength ),num_inference_steps )
        t_start =max (num_inference_steps -init_timestep ,0 )
        timesteps =self .scheduler .timesteps [t_start *self .scheduler .order :]
        return timesteps ,num_inference_steps -t_start
    @property
    def guidance_scale1 (self ):
        return self ._guidance_scale1
    @property
    def guidance_scale2 (self ):
        return self ._guidance_scale2
    @property
    def num_timesteps (self ):
        return self ._num_timesteps
    @torch .no_grad ()
    def __call__ (
    self ,
    ref_image :Union [PIL .Image .Image ,List [PIL .Image .Image ],torch .FloatTensor ],
    ref_concat_image :Union [PIL .Image .Image ,List [PIL .Image .Image ],torch .FloatTensor ],
    id_prompts :Union [PIL .Image .Image ,List [PIL .Image .Image ],torch .FloatTensor ],
    task_id_input :torch .Tensor =None ,
    height :int =512 ,
    width :int =512 ,
    num_frames :Optional [int ]=None ,
    num_inference_steps :int =25 ,
    min_guidance_scale =1.0 ,
    max_guidance_scale =3.0 ,
    noise_aug_strength :int =0.02 ,
    decode_chunk_size :Optional [int ]=None ,
    num_videos_per_prompt :Optional [int ]=1 ,
    generator :Optional [Union [torch .Generator ,List [torch .Generator ]]]=None ,
    latents :Optional [torch .FloatTensor ]=None ,
    output_type :Optional [str ]="pil",
    callback_on_step_end :Optional [Callable [[int ,int ,Dict ],None ]]=None ,
    callback_on_step_end_tensor_inputs :List [str ]=["latents"],
    return_dict :bool =True ,
    do_classifier_free_guidance :bool =True ,
    overlap =7 ,
    frames_per_batch =14 ,
    i2i_noise_strength =1.0 ,
    ):
        r
        height =height or self .unet .config .sample_size *self .vae_scale_factor
        width =width or self .unet .config .sample_size *self .vae_scale_factor
        num_frames =num_frames if num_frames is not None else self .unet .config .num_frames
        decode_chunk_size =decode_chunk_size if decode_chunk_size is not None else num_frames
        self .check_inputs (ref_image ,height ,width )
        if isinstance (ref_image ,PIL .Image .Image ):
            batch_size =1
        elif isinstance (ref_image ,list ):
            batch_size =len (ref_image )
        else :
            if len (ref_image .shape )==4 :
                batch_size =1
            else :
                batch_size =ref_image .shape [0 ]
        device =self ._execution_device
        image_embeddings =torch .cat ([torch .zeros_like (id_prompts ),id_prompts ],dim =0 )if do_classifier_free_guidance else id_prompts
        needs_upcasting =(self .vae .dtype ==torch .float16 or self .vae .dtype ==torch .bfloat16 )and self .vae .config .force_upcast
        vae_dtype =self .vae .dtype
        if needs_upcasting :
            self .vae .to (dtype =torch .float32 )
        ref_image_tensor =ref_image .to (
        dtype =self .vae .dtype ,device =self .vae .device
        )
        chunk_size =20
        ref_image_latents =[]
        for chunk_idx in range ((ref_image_tensor .shape [0 ]//chunk_size )+1 ):
            if chunk_idx *chunk_size >=num_frames :break
            ref_image_latent =self .vae .encode (ref_image_tensor [chunk_idx *chunk_size :(chunk_idx +1 )*chunk_size ]).latent_dist .mean
            ref_image_latents .append (ref_image_latent )
        ref_image_latents =torch .cat (ref_image_latents ,dim =0 )
        ref_image_latents =ref_image_latents *0.18215
        noise =randn_tensor (
        ref_image_tensor .shape ,
        generator =generator ,
        device =self .vae .device ,
        dtype =self .vae .dtype )
        ref_image_tensor =ref_image_tensor +noise_aug_strength *noise
        image_latents =[]
        for chunk_idx in range ((ref_image_tensor .shape [0 ]//chunk_size )+1 ):
            if chunk_idx *chunk_size >=num_frames :break
            image_latent =self ._encode_vae_image (
            ref_image_tensor [chunk_idx *chunk_size :(chunk_idx +1 )*chunk_size ],
            device =device ,
            num_videos_per_prompt =num_videos_per_prompt ,
            do_classifier_free_guidance =do_classifier_free_guidance ,
            )
            image_latents .append (image_latent )
        image_latents =torch .cat (image_latents ,dim =1 )
        image_latents =image_latents .to (image_embeddings .dtype )
        ref_image_latents =ref_image_latents .to (image_embeddings .dtype )
        if needs_upcasting :
            self .vae .to (dtype =vae_dtype )
        if ref_concat_image is not None :
            ref_concat_tensor =ref_concat_image .to (
            dtype =self .vae .dtype ,device =self .vae .device
            )
            ref_concat_tensor =self .vae .encode (ref_concat_tensor .unsqueeze (0 )).latent_dist .mode ()
            ref_concat_tensor =ref_concat_tensor .unsqueeze (0 ).repeat (1 ,num_frames ,1 ,1 ,1 )
            ref_concat_tensor =torch .cat ([torch .zeros_like (ref_concat_tensor ),ref_concat_tensor ])if do_classifier_free_guidance else ref_concat_tensor
            ref_concat_tensor =ref_concat_tensor .to (image_embeddings )
        else :
            ref_concat_tensor =torch .zeros_like (image_latents )
        added_time_ids =self ._get_add_time_ids (
        task_id_input ,
        image_embeddings .dtype ,
        batch_size ,
        num_videos_per_prompt ,
        do_classifier_free_guidance ,
        )
        added_time_ids =added_time_ids .to (device ,dtype =self .unet .dtype )
        self .scheduler .set_timesteps (num_inference_steps ,device =device )
        timesteps ,num_inference_steps =self .get_timesteps (num_inference_steps ,i2i_noise_strength ,device )
        latent_timestep =timesteps [:1 ].repeat (batch_size *num_videos_per_prompt )
        num_channels_latents =self .unet .config .in_channels
        latents =self .prepare_latents (
        batch_size *num_videos_per_prompt ,
        num_frames ,
        num_channels_latents ,
        height ,
        width ,
        image_embeddings .dtype ,
        device ,
        generator ,
        latents ,
        ref_image_latents ,
        timestep =latent_timestep
        )
        guidance_scale =torch .linspace (
        min_guidance_scale ,
        max_guidance_scale ,
        num_inference_steps )
        guidance_scale1 =guidance_scale .to (device ,latents .dtype )
        guidance_scale2 =guidance_scale .to (device ,latents .dtype )
        self ._guidance_scale1 =guidance_scale1
        self ._guidance_scale2 =guidance_scale2
        latents_all =latents
        num_warmup_steps =len (timesteps )-num_inference_steps *self .scheduler .order
        self ._num_timesteps =len (timesteps )
        shift =0
        with self .progress_bar (total =num_inference_steps )as progress_bar :
            for i ,t in enumerate (timesteps ):
                pred_latents =torch .zeros_like (
                latents_all ,
                dtype =self .unet .dtype ,
                )
                counter =torch .zeros (
                (latents_all .shape [0 ],num_frames ,1 ,1 ,1 ),
                dtype =self .unet .dtype ,
                ).to (device =latents_all .device )
                for batch ,index_start in enumerate (range (0 ,num_frames ,frames_per_batch -overlap *(i <3 ))):
                    self .scheduler ._step_index =None
                    index_start -=shift
                    def indice_slice (tensor ,idx_list ):
                        tensor_list =[]
                        for idx in idx_list :
                            idx =idx %tensor .shape [1 ]
                            tensor_list .append (tensor [:,idx ])
                        return torch .stack (tensor_list ,1 )
                    idx_list =list (range (index_start ,index_start +frames_per_batch ))
                    latents =indice_slice (latents_all ,idx_list )
                    image_latents_input =indice_slice (image_latents ,idx_list )
                    image_embeddings_input =indice_slice (image_embeddings ,idx_list )
                    ref_concat_tensor_input =indice_slice (ref_concat_tensor ,idx_list )
                    latent_model_input =torch .cat ([latents ]*2 )if do_classifier_free_guidance else latents
                    latent_model_input =self .scheduler .scale_model_input (latent_model_input ,t )
                    latent_model_input =torch .cat ([
                    latent_model_input ,
                    image_latents_input ,
                    ref_concat_tensor_input ],dim =2 )
                    noise_pred =self .unet (
                    latent_model_input ,
                    t ,
                    encoder_hidden_states =image_embeddings_input .flatten (0 ,1 ),
                    added_time_ids =added_time_ids ,
                    return_dict =False ,
                    )[0 ]
                    if do_classifier_free_guidance :
                        noise_pred_uncond ,noise_pred_cond =noise_pred .chunk (3 )
                        noise_pred =noise_pred_uncond +self .guidance_scale1 [i ]*(noise_pred_cond -noise_pred_uncond )
                    latents =self .scheduler .step (noise_pred ,t .to (self .unet .dtype ),latents ).prev_sample
                    if callback_on_step_end is not None :
                        callback_kwargs ={}
                        for k in callback_on_step_end_tensor_inputs :
                            callback_kwargs [k ]=locals ()[k ]
                        callback_outputs =callback_on_step_end (self ,i ,t ,callback_kwargs )
                        latents =callback_outputs .pop ("latents",latents )
                    for iii in range (frames_per_batch ):
                        p =(index_start +iii )%pred_latents .shape [1 ]
                        pred_latents [:,p ]+=latents [:,iii ]*min (iii +1 ,frames_per_batch -iii )
                        counter [:,p ]+=1 *min (iii +1 ,frames_per_batch -iii )
                shift +=overlap
                shift =shift %frames_per_batch
                pred_latents =pred_latents /counter
                latents_all =pred_latents
                if i ==len (timesteps )-1 or ((i +1 )>num_warmup_steps and (i +1 )%self .scheduler .order ==0 ):
                    progress_bar .update ()
        latents =latents_all
        if not output_type =="latent":
            if needs_upcasting :
                self .vae .to (dtype =vae_dtype )
            frames =self .decode_latents (latents ,num_frames ,decode_chunk_size )
        else :
            frames =latents
        self .maybe_free_model_hooks ()
        if not return_dict :
            return frames
        return LQ2VideoSVDPipelineOutput (frames =frames ,latents =latents )
def _resize_with_antialiasing (input ,size ,interpolation ="bicubic",align_corners =True ):
    h ,w =input .shape [-2 :]
    factors =(h /size [0 ],w /size [1 ])
    sigmas =(
    max ((factors [0 ]-1.0 )/2.0 ,0.001 ),
    max ((factors [1 ]-1.0 )/2.0 ,0.001 ),
    )
    ks =int (max (2.0 *2 *sigmas [0 ],3 )),int (max (2.0 *2 *sigmas [1 ],3 ))
    if (ks [0 ]%2 )==0 :
        ks =ks [0 ]+1 ,ks [1 ]
    if (ks [1 ]%2 )==0 :
        ks =ks [0 ],ks [1 ]+1
    input =_gaussian_blur2d (input ,ks ,sigmas )
    output =torch .nn .functional .interpolate (input ,size =size ,mode =interpolation ,align_corners =align_corners )
    return output
def _compute_padding (kernel_size ):
    if len (kernel_size )<2 :
        raise AssertionError (kernel_size )
    computed =[k -1 for k in kernel_size ]
    out_padding =2 *len (kernel_size )*[0 ]
    for i in range (len (kernel_size )):
        computed_tmp =computed [-(i +1 )]
        pad_front =computed_tmp //2
        pad_rear =computed_tmp -pad_front
        out_padding [2 *i +0 ]=pad_front
        out_padding [2 *i +1 ]=pad_rear
    return out_padding
def _filter2d (input ,kernel ):
    b ,c ,h ,w =input .shape
    tmp_kernel =kernel [:,None ,...].to (device =input .device ,dtype =input .dtype )
    tmp_kernel =tmp_kernel .expand (-1 ,c ,-1 ,-1 )
    height ,width =tmp_kernel .shape [-2 :]
    padding_shape :list [int ]=_compute_padding ([height ,width ])
    input =torch .nn .functional .pad (input ,padding_shape ,mode ="reflect")
    tmp_kernel =tmp_kernel .reshape (-1 ,1 ,height ,width )
    input =input .view (-1 ,tmp_kernel .size (0 ),input .size (-2 ),input .size (-1 ))
    output =torch .nn .functional .conv2d (input ,tmp_kernel ,groups =tmp_kernel .size (0 ),padding =0 ,stride =1 )
    out =output .view (b ,c ,h ,w )
    return out
def _gaussian (window_size :int ,sigma ):
    if isinstance (sigma ,float ):
        sigma =torch .tensor ([[sigma ]])
    batch_size =sigma .shape [0 ]
    x =(torch .arange (window_size ,device =sigma .device ,dtype =sigma .dtype )-window_size //2 ).expand (batch_size ,-1 )
    if window_size %2 ==0 :
        x =x +0.5
    gauss =torch .exp (-x .pow (2.0 )/(2 *sigma .pow (2.0 )))
    return gauss /gauss .sum (-1 ,keepdim =True )
def _gaussian_blur2d (input ,kernel_size ,sigma ):
    if isinstance (sigma ,tuple ):
        sigma =torch .tensor ([sigma ],dtype =input .dtype )
    else :
        sigma =sigma .to (dtype =input .dtype )
    ky ,kx =int (kernel_size [0 ]),int (kernel_size [1 ])
    bs =sigma .shape [0 ]
    kernel_x =_gaussian (kx ,sigma [:,1 ].view (bs ,1 ))
    kernel_y =_gaussian (ky ,sigma [:,0 ].view (bs ,1 ))
    out_x =_filter2d (input ,kernel_x [...,None ,:])
    out =_filter2d (out_x ,kernel_y [...,None ])
    return out