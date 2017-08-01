
const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

kernel void exec1(__read_only image2d_t in, write_only image2d_t out) {
    
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
        
    uint4 tap = read_imageui(in, sampler, (int2)(x,y));
    
    uint param = (tap.x + tap.y + tap.z) / 3;
    
    tap.x = param;
    tap.y = param;
    tap.z = param;
    
    write_imageui(out, (int2)(x,y), tap.xyzw);
    
}

kernel void exec2(__read_only image2d_t in, write_only image2d_t out) {
    
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    
    uint4 tap = read_imageui(in, sampler, (int2)(x,y));
    
    uint param = (uint) (0.299 * tap.x + 0.587 * tap.y + 0.114 * tap.z);
    
    tap.x = param;
    tap.y = param;
    tap.z = param;
    
    write_imageui(out, (int2)(x,y), tap.xyzw);
}