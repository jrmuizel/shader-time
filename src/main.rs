extern crate gleam;
extern crate glutin;
extern crate libc;
extern crate euclid;

use euclid::{Transform3D, Vector3D};

use gleam::gl::Gl;
use gleam::gl;
use glutin::GlContext;

use std::rc::Rc;

use gleam::gl::GLuint;
use gleam::gl::ErrorCheckingGl;



/* It was already known that the efficiency gains from client storage only materialize if you
   follow certain restrictions:
   - The textures need to use the TEXTURE_RECTANGLE_ARB texture target.
   - The textures' format, internalFormat and type need to be chosen from a small list of
     supported configurations. Unsupported configurations will trigger format conversions on the CPU.
   - The GL_TEXTURE_STORAGE_HINT_APPLE may need to be set to shared or cached. -
     glTextureRangeAPPLE may or may not make a difference.

 It now appears that the stride alignment is another requirement: When uploading textures which
 otherwise comply with the above requirements, the Intel driver will still make copies using the
 CPU if the texture's stride is not 32-byte aligned. These CPU copies are reflected in a high CPU
 usage (as observed in Activity Monitor) and they show up in profiles as time spent inside
 _platform_memmove under glrUpdateTexture.

 */

struct Options {
    pbo: bool,
    client_storage: bool,
    texture_array: bool,
    texture_storage: bool,
    swizzle: bool,
}
use std::time::Instant;


fn init_shader_program(gl: &Rc<Gl>, vs_source: &[u8], fs_source: &[u8]) -> gl::GLuint {
    let now = Instant::now();


    let vertex_shader = load_shader(gl, gl::VERTEX_SHADER, vs_source);
    let fragment_shader = load_shader(gl, gl::FRAGMENT_SHADER, fs_source);
    let shader_program = gl.create_program();
    gl.attach_shader(shader_program, vertex_shader);
    gl.attach_shader(shader_program, fragment_shader);
    gl.link_program(shader_program);

    let mut link_status = [0];
    unsafe {
        gl.get_program_iv(shader_program, gl::LINK_STATUS, &mut link_status);
            if link_status[0] == 0 {
                println!("LINK: {}", gl.get_program_info_log(shader_program));
            }

    }

    // it prints '2'
    println!("{}", now.elapsed().as_secs_f32());
    shader_program
}

struct Buffers {
    position: GLuint,
    texture_coord: GLuint,
    indices: GLuint
}

fn init_buffers(gl: &Rc<gl::Gl>, texture_rectangle: bool, texture_width: i32, texture_height: i32) -> Buffers {
    let position_buffer = gl.gen_buffers(1)[0];

    gl.bind_buffer(gl::ARRAY_BUFFER, position_buffer);

    let positions = [
        // Front face
        -1.0f32, -1.0,  1.0,
        1.0, -1.0,  1.0,
        1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0,

        // Back face
        -1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        1.0,  1.0, -1.0,
        1.0, -1.0, -1.0,

        // Top face
        -1.0,  1.0, -1.0,
        -1.0,  1.0,  1.0,
        1.0,  1.0,  1.0,
        1.0,  1.0, -1.0,

        // Bottom face
        -1.0, -1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0, -1.0,  1.0,
        -1.0, -1.0,  1.0,

        // Right face
        1.0, -1.0, -1.0,
        1.0,  1.0, -1.0,
        1.0,  1.0,  1.0,
        1.0, -1.0,  1.0,

        // Left face
        -1.0, -1.0, -1.0,
        -1.0, -1.0,  1.0,
        -1.0,  1.0,  1.0,
        -1.0,  1.0, -1.0,
    ];


    gl.buffer_data_untyped(gl::ARRAY_BUFFER, std::mem::size_of_val(&positions) as isize, positions.as_ptr() as *const libc::c_void, gl::STATIC_DRAW);


    let texture_coord_buffer = gl.gen_buffers(1)[0];

    gl.bind_buffer(gl::ARRAY_BUFFER, texture_coord_buffer);

    let width = if texture_rectangle { texture_width as f32 } else { 1.0 };
    let height = if texture_rectangle { texture_height as f32 } else { 1.0 };

    let texture_coordinates = [
        // Front
        0.0f32,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
        // Back
        0.0,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
        // Top
        0.0,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
        // Bottom
        0.0,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
        // Right
        0.0,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
        // Left
        0.0,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
    ];

    gl.buffer_data_untyped(gl::ARRAY_BUFFER, std::mem::size_of_val(&texture_coordinates) as isize, texture_coordinates.as_ptr() as *const libc::c_void, gl::STATIC_DRAW);

    // Build the element array buffer; this specifies the indices
    // into the vertex arrays for each face's vertices.

    let index_buffer = gl.gen_buffers(1)[0];

    gl.bind_buffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer);

    // This array defines each face as two triangles, using the
    // indices into the vertex array to specify each triangle's
    // position.

    let indices = [
        0u16,  1,  2,      0,  2,  3,    // front
        4,  5,  6,      4,  6,  7,    // back
        8,  9,  10,     8,  10, 11,   // top
        12, 13, 14,     12, 14, 15,   // bottom
        16, 17, 18,     16, 18, 19,   // right
        20, 21, 22,     20, 22, 23,   // left
    ];

    // Now send the element array to GL

    gl.buffer_data_untyped(gl::ELEMENT_ARRAY_BUFFER, std::mem::size_of_val(&indices) as isize, indices.as_ptr() as *const libc::c_void, gl::STATIC_DRAW);


    Buffers {
        position: position_buffer,
        texture_coord: texture_coord_buffer,
        indices: index_buffer,
    }
}

struct Image {
    data: Vec<u8>,
    width: i32,
    height: i32
}

fn rgba_to_bgra(buf: &mut [u8]) {
    assert!(buf.len() % 4 == 0);
    let mut i = 0;
    while i < buf.len() {
        let r = buf[i];
        let g = buf[i+1];
        let b = buf[i+2];
        let a = buf[i+3];
        buf[i] = b;
        buf[i+1] = g;
        buf[i+2] = r;
        buf[i+3] = a;
        i += 4;
    }
}

fn make_blue(buf: &mut [u8]) {
    assert!(buf.len() % 4 == 0);
    let mut i = 0;
    while i < buf.len() {
        buf[i] = 0xff;
        buf[i+1] = 0;
        buf[i+2] = 0;
        buf[i+3] = 0xff;
        i += 4;
    }
}

fn make_yellow(buf: &mut [u8]) {
    assert!(buf.len() % 4 == 0);
    let mut i = 0;
    while i < buf.len() {
        buf[i] = 0;
        buf[i+1] = 0xff;
        buf[i+2] = 0xff;
        buf[i+3] = 0xff;
        i += 4;
    }
}

fn paint_square(image: &mut Image) {
    let width = image.width as usize;
    for i in 1024..2048 {
        make_yellow(&mut image.data[i*width..(i*width + 512)]);
    }

}

fn paint_square2(image: &mut Image) {
    let width = image.width as usize;
    for i in 1024..2048 {
        make_blue(&mut image.data[i*width..(i*width + 512)]);
    }

}

fn bpp(format: GLuint) -> i32 {
    match format {
        gl::UNSIGNED_INT_8_8_8_8_REV => 4,
        gl::UNSIGNED_BYTE => 4,
        gl::FLOAT => 16,
        gl::INT => 16,
        gl::UNSIGNED_INT => 16,
        _ => panic!()
    }
}



fn load_image(format: GLuint) -> Image
{
    if true {
        let width: i32 = 4096;
        let height: i32 = 8192;
        return Image { data: vec![0; (width * height * bpp(format)) as usize], width, height }
    }
    let decoder = png::Decoder::new(std::fs::File::open("cubetexture.png").unwrap());
    let (info, mut reader) = decoder.read_info().unwrap();
    // Allocate the output buffer.
    let mut buf = vec![0; info.buffer_size()];
    // Read the next frame. Currently this function should only called once.
    // The default options
    reader.next_frame(&mut buf).unwrap();

    rgba_to_bgra(&mut buf);

    //make_red(&mut buf);
    //make_yellow(&mut buf);

    Image { data: buf, width: info.width as i32, height: info.height as i32}
}

struct Texture {
    id: GLuint,
    pbo: Option<GLuint>,
}


fn load_texture(gl: &Rc<gl::Gl>, image: &Image, target: GLuint, internal_format: GLuint, src_format: GLuint, src_type: GLuint, options: &Options) -> Texture {
    let texture = gl.gen_textures(1)[0];

    gl.bind_texture(target, texture);

    let level = 0;
    let border = 0;

    if options.client_storage {
        //gl.texture_range_apple(target, &image.data[..]);

        // both of these seem to work ok on Intel
        let storage = gl::STORAGE_SHARED_APPLE;
        let storage = gl::STORAGE_CACHED_APPLE;
        gl.tex_parameter_i(target, gl::TEXTURE_STORAGE_HINT_APPLE, storage as gl::GLint);
        gl.pixel_store_i(gl::UNPACK_CLIENT_STORAGE_APPLE, true as gl::GLint);

        // this may not be needed
        gl.pixel_store_i(gl::UNPACK_ROW_LENGTH, 0);
    }

    if options.texture_array {
        if options.texture_storage {
            gl.tex_storage_3d(target, 1, internal_format, image.width as i32, image.height as i32, 1);
        } else {
            gl.tex_image_3d(target, level, internal_format as i32, image.width, image.height, 1, border, src_format, src_type, if options.pbo { None } else { Some(&image.data[..]) });
        }
    } else {
        if options.texture_storage {
            gl.tex_storage_2d(target, 1, internal_format, image.width as i32, image.height as i32);
        } else {
            gl.tex_image_2d(target, level, internal_format as i32, image.width, image.height, border, src_format, src_type, if options.pbo { None } else { Some(&image.data[..]) });
        }
    }

    // Rectangle textures has its limitations compared to using POT textures, for example,
    // Rectangle textures can't use mipmap filtering
    gl.tex_parameter_i(target, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);

    // Rectangle textures can't use the GL_REPEAT warp mode
    gl.tex_parameter_i(target, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as gl::GLint);
    gl.tex_parameter_i(target, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::GLint);
    if options.swizzle {
        //let components = [gl::RED, gl::GREEN, gl::BLUE, gl::ALPHA];
        let components = [gl::BLUE, gl::GREEN, gl::RED, gl::ALPHA];
        gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_R, components[0] as i32);
        gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_G, components[1] as i32);
        gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_B, components[2] as i32);
        gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_A, components[3] as i32);
    }

    let pbo = if options.pbo {
        let id = gl.gen_buffers(1)[0];
        // WebRender on Mac uses DYNAMIC_DRAW
        gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, id);
        gl.buffer_data_untyped(gl::PIXEL_UNPACK_BUFFER, (image.width * image.height * bpp(src_type)) as _, image.data[..].as_ptr() as *const libc::c_void, gl::DYNAMIC_DRAW);

        if options.texture_array {
            gl.tex_sub_image_3d_pbo(
                target,
                level,
                0,
                0,
                0,
                image.width,
                image.height,
                1,
                src_format,
                src_type,
                0,
            );
        } else {
            gl.tex_sub_image_2d_pbo(
                target,
                level,
                0,
                0,
                image.width,
                image.height,
                src_format,
                src_type,
                0,
            );
        }
        gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);
        Some(id)
    } else {
        None
    };
    Texture { id: texture, pbo }
}


fn load_shader(gl: &Rc<Gl>, shader_type: gl::GLenum, source: &[u8]) -> gl::GLuint {
    let shader = gl.create_shader(shader_type);
    gl.shader_source(shader, &[source]);
    gl.compile_shader(shader);
    let mut status = [0];
    unsafe { gl.get_shader_iv(shader, gl::COMPILE_STATUS, &mut status); }
    if status[0] == 0 {
        println!("{}", gl.get_shader_info_log(shader));
        panic!();
    }
    return shader;
}


fn main() {
    
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(1024, 768);
    let context = glutin::ContextBuilder::new()
        .with_vsync(true)
        .with_gl(glutin::GlRequest::GlThenGles {
        opengl_version: (3, 2),
        opengles_version: (3, 0),
    });

    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

    unsafe {
        gl_window.make_current().unwrap();
    }

    let options = Options { pbo: false, client_storage: false, texture_array: false, texture_storage: false, swizzle: false };

    let texture_rectangle = true;
    let apple_format = true; // on Intel it looks like we don't need this particular format

    let texture_target = if texture_rectangle { gl::TEXTURE_RECTANGLE_ARB } else { gl::TEXTURE_2D };
    let texture_target = if options.texture_array { gl::TEXTURE_2D_ARRAY} else { texture_target };

    //let texture_internal_format = gl::RGBA32UI;
    //let texture_internal_format = gl::RGBA32F;
    let texture_internal_format = gl::RGBA8;

    let mut texture_src_format = if apple_format { gl::BGRA } else { gl::RGBA };
    let mut texture_src_type = if apple_format { gl::UNSIGNED_INT_8_8_8_8_REV } else { gl::UNSIGNED_BYTE };

    // adjust type and format to match internal format
    if texture_internal_format == gl::RGBA32UI {
        texture_src_type = gl::UNSIGNED_INT;
        texture_src_format = gl::RGBA_INTEGER;
    } else if texture_internal_format == gl::RGBA32F {
        texture_src_format = gl::RGBA;
        texture_src_type = gl::FLOAT;
    }


    let vs_source = br#"
    #version 150
    // ps_text_run
    #define WR_VERTEX_SHADER
    #define WR_MAX_VERTEX_TEXTURE_WIDTH 1024U
    #define WR_FEATURE_
    /* This Source Code Form is subject to the terms of the Mozilla Public
     * License, v. 2.0. If a copy of the MPL was not distributed with this
     * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
    
    /* This Source Code Form is subject to the terms of the Mozilla Public
     * License, v. 2.0. If a copy of the MPL was not distributed with this
     * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
    
    #ifdef WR_FEATURE_PIXEL_LOCAL_STORAGE
    // For now, we need both extensions here, in order to initialize
    // the PLS to the current framebuffer color. In future, we can
    // possibly remove that requirement, or at least support the
    // other framebuffer fetch extensions that provide the same
    // functionality.
    #extension GL_EXT_shader_pixel_local_storage : require
    #extension GL_ARM_shader_framebuffer_fetch : require
    #endif
    
    #ifdef WR_FEATURE_TEXTURE_EXTERNAL
    // Please check https://www.khronos.org/registry/OpenGL/extensions/OES/OES_EGL_image_external_essl3.txt
    // for this extension.
    #extension GL_OES_EGL_image_external_essl3 : require
    #endif
    
    #ifdef WR_FEATURE_ADVANCED_BLEND
    #extension GL_KHR_blend_equation_advanced : require
    #endif
    
    #ifdef WR_FEATURE_DUAL_SOURCE_BLENDING
    #ifdef GL_ES
    #extension GL_EXT_blend_func_extended : require
    #else
    #extension GL_ARB_explicit_attrib_location : require
    #endif
    #endif
    
    /* This Source Code Form is subject to the terms of the Mozilla Public
     * License, v. 2.0. If a copy of the MPL was not distributed with this
     * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
    
    #if defined(GL_ES)
        #if GL_ES == 1
            #ifdef GL_FRAGMENT_PRECISION_HIGH
            precision highp sampler2DArray;
            #else
            precision mediump sampler2DArray;
            #endif
    
            // Sampler default precision is lowp on mobile GPUs.
            // This causes RGBA32F texture data to be clamped to 16 bit floats on some GPUs (e.g. Mali-T880).
            // Define highp precision macro to allow lossless FLOAT texture sampling.
            #define HIGHP_SAMPLER_FLOAT highp
    
            // Default int precision in GLES 3 is highp (32 bits) in vertex shaders
            // and mediump (16 bits) in fragment shaders. If an int is being used as
            // a texel address in a fragment shader it, and therefore requires > 16
            // bits, it must be qualified with this.
            #define HIGHP_FS_ADDRESS highp
    
            // texelFetchOffset is buggy on some Android GPUs (see issue #1694).
            // Fallback to texelFetch on mobile GPUs.
            #define TEXEL_FETCH(sampler, position, lod, offset) texelFetch(sampler, position + offset, lod)
        #else
            #define HIGHP_SAMPLER_FLOAT
            #define HIGHP_FS_ADDRESS
            #define TEXEL_FETCH(sampler, position, lod, offset) texelFetchOffset(sampler, position, lod, offset)
        #endif
    #else
        #define HIGHP_SAMPLER_FLOAT
        #define HIGHP_FS_ADDRESS
        #define TEXEL_FETCH(sampler, position, lod, offset) texelFetchOffset(sampler, position, lod, offset)
    #endif
    
    #ifdef WR_VERTEX_SHADER
        #define varying out
    #endif
    
    #ifdef WR_FRAGMENT_SHADER
        precision highp float;
        #define varying in
    #endif
    
    #if defined(WR_FEATURE_TEXTURE_EXTERNAL) || defined(WR_FEATURE_TEXTURE_RECT) || defined(WR_FEATURE_TEXTURE_2D)
    #define TEX_SAMPLE(sampler, tex_coord) texture(sampler, tex_coord.xy)
    #else
    #define TEX_SAMPLE(sampler, tex_coord) texture(sampler, tex_coord)
    #endif
    
    //======================================================================================
    // Vertex shader attributes and uniforms
    //======================================================================================
    #ifdef WR_VERTEX_SHADER
        // A generic uniform that shaders can optionally use to configure
        // an operation mode for this batch.
        uniform int uMode;
    
        // Uniform inputs
        uniform mat4 uTransform;       // Orthographic projection
    
        // Attribute inputs
        in vec3 aPosition;
    
        // get_fetch_uv is a macro to work around a macOS Intel driver parsing bug.
        // TODO: convert back to a function once the driver issues are resolved, if ever.
        // https://github.com/servo/webrender/pull/623
        // https://github.com/servo/servo/issues/13953
        // Do the division with unsigned ints because that's more efficient with D3D
        #define get_fetch_uv(i, vpi)  ivec2(int(vpi * (uint(i) % (WR_MAX_VERTEX_TEXTURE_WIDTH/vpi))), int(uint(i) / (WR_MAX_VERTEX_TEXTURE_WIDTH/vpi)))
    #endif
    
    //======================================================================================
    // Fragment shader attributes and uniforms
    //======================================================================================
    #ifdef WR_FRAGMENT_SHADER
        // Uniform inputs
    
        #ifdef WR_FEATURE_PIXEL_LOCAL_STORAGE
            // Define the storage class of the pixel local storage.
            // If defined as writable, it's a compile time error to
            // have a normal fragment output variable declared.
            #if defined(PLS_READONLY)
                #define PLS_BLOCK __pixel_local_inEXT
            #elif defined(PLS_WRITEONLY)
                #define PLS_BLOCK __pixel_local_outEXT
            #else
                #define PLS_BLOCK __pixel_localEXT
            #endif
    
            // The structure of pixel local storage. Right now, it's
            // just the current framebuffer color. In future, we have
            // (at least) 12 bytes of space we can store extra info
            // here (such as clip mask values).
            PLS_BLOCK FrameBuffer {
                layout(rgba8) highp vec4 color;
            } PLS;
    
            #ifndef PLS_READONLY
            // Write the output of a fragment shader to PLS. Applies
            // premultipled alpha blending by default, since the blender
            // is disabled when PLS is active.
            // TODO(gw): Properly support alpha blend mode for webgl / canvas.
            void write_output(vec4 color) {
                PLS.color = color + PLS.color * (1.0 - color.a);
            }
    
            // Write a raw value straight to PLS, if the fragment shader has
            // already applied blending.
            void write_output_raw(vec4 color) {
                PLS.color = color;
            }
            #endif
    
            #ifndef PLS_WRITEONLY
            // Retrieve the current framebuffer color. Useful in conjunction with
            // the write_output_raw function.
            vec4 get_current_framebuffer_color() {
                return PLS.color;
            }
            #endif
        #else
            // Fragment shader outputs
            #ifdef WR_FEATURE_ADVANCED_BLEND
                layout(blend_support_all_equations) out;
            #endif
    
            #ifdef WR_FEATURE_DUAL_SOURCE_BLENDING
                layout(location = 0, index = 0) out vec4 oFragColor;
                layout(location = 0, index = 1) out vec4 oFragBlend;
            #else
                out vec4 oFragColor;
            #endif
    
            // Write an output color in normal (non-PLS) shaders.
            void write_output(vec4 color) {
                oFragColor = color;
            }
        #endif
    
        #define EPSILON                     0.0001
    
        // "Show Overdraw" color. Premultiplied.
        #define WR_DEBUG_OVERDRAW_COLOR     vec4(0.110, 0.077, 0.027, 0.125)
    
        float distance_to_line(vec2 p0, vec2 perp_dir, vec2 p) {
            vec2 dir_to_p0 = p0 - p;
            return dot(normalize(perp_dir), dir_to_p0);
        }
    
        /// Find the appropriate half range to apply the AA approximation over.
        /// This range represents a coefficient to go from one CSS pixel to half a device pixel.
        float compute_aa_range(vec2 position) {
            // The constant factor is chosen to compensate for the fact that length(fw) is equal
            // to sqrt(2) times the device pixel ratio in the typical case. 0.5/sqrt(2) = 0.35355.
            //
            // This coefficient is chosen to ensure that any sample 0.5 pixels or more inside of
            // the shape has no anti-aliasing applied to it (since pixels are sampled at their center,
            // such a pixel (axis aligned) is fully inside the border). We need this so that antialiased
            // curves properly connect with non-antialiased vertical or horizontal lines, among other things.
            //
            // Lines over a half-pixel away from the pixel center *can* intersect with the pixel square;
            // indeed, unless they are horizontal or vertical, they are guaranteed to. However, choosing
            // a nonzero area for such pixels causes noticeable artifacts at the junction between an anti-
            // aliased corner and a straight edge.
            //
            // We may want to adjust this constant in specific scenarios (for example keep the principled
            // value for straight edges where we want pixel-perfect equivalence with non antialiased lines
            // when axis aligned, while selecting a larger and smoother aa range on curves).
            return 0.35355 * length(fwidth(position));
        }
    
        /// Return the blending coefficient for distance antialiasing.
        ///
        /// 0.0 means inside the shape, 1.0 means outside.
        ///
        /// This cubic polynomial approximates the area of a 1x1 pixel square under a
        /// line, given the signed Euclidean distance from the center of the square to
        /// that line. Calculating the *exact* area would require taking into account
        /// not only this distance but also the angle of the line. However, in
        /// practice, this complexity is not required, as the area is roughly the same
        /// regardless of the angle.
        ///
        /// The coefficients of this polynomial were determined through least-squares
        /// regression and are accurate to within 2.16% of the total area of the pixel
        /// square 95% of the time, with a maximum error of 3.53%.
        ///
        /// See the comments in `compute_aa_range()` for more information on the
        /// cutoff values of -0.5 and 0.5.
        float distance_aa(float aa_range, float signed_distance) {
            float dist = 0.5 * signed_distance / aa_range;
            if (dist <= -0.5 + EPSILON)
                return 1.0;
            if (dist >= 0.5 - EPSILON)
                return 0.0;
            return 0.5 + dist * (0.8431027 * dist * dist - 1.14453603);
        }
    
        /// Component-wise selection.
        ///
        /// The idea of using this is to ensure both potential branches are executed before
        /// selecting the result, to avoid observable timing differences based on the condition.
        ///
        /// Example usage: color = if_then_else(LessThanEqual(color, vec3(0.5)), vec3(0.0), vec3(1.0));
        ///
        /// The above example sets each component to 0.0 or 1.0 independently depending on whether
        /// their values are below or above 0.5.
        ///
        /// This is written as a macro in order to work with vectors of any dimension.
        ///
        /// Note: Some older android devices don't support mix with bvec. If we ever run into them
        /// the only option we have is to polyfill it with a branch per component.
        #define if_then_else(cond, then_branch, else_branch) mix(else_branch, then_branch, cond)
    #endif
    
    //======================================================================================
    // Shared shader uniforms
    //======================================================================================
    #ifdef WR_FEATURE_TEXTURE_2D
    uniform sampler2D sColor0;
    uniform sampler2D sColor1;
    uniform sampler2D sColor2;
    #elif defined WR_FEATURE_TEXTURE_RECT
    uniform sampler2DRect sColor0;
    uniform sampler2DRect sColor1;
    uniform sampler2DRect sColor2;
    #elif defined WR_FEATURE_TEXTURE_EXTERNAL
    uniform samplerExternalOES sColor0;
    uniform samplerExternalOES sColor1;
    uniform samplerExternalOES sColor2;
    #else
    uniform sampler2DArray sColor0;
    uniform sampler2DArray sColor1;
    uniform sampler2DArray sColor2;
    #endif
    
    #ifdef WR_FEATURE_DITHERING
    uniform sampler2D sDither;
    #endif
    
    //======================================================================================
    // Interpolator definitions
    //======================================================================================
    
    //======================================================================================
    // VS only types and UBOs
    //======================================================================================
    
    //======================================================================================
    // VS only functions
    //======================================================================================
    /* This Source Code Form is subject to the terms of the Mozilla Public
     * License, v. 2.0. If a copy of the MPL was not distributed with this
     * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
    
    /* This Source Code Form is subject to the terms of the Mozilla Public
     * License, v. 2.0. If a copy of the MPL was not distributed with this
     * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
    
    struct RectWithSize {
        vec2 p0;
        vec2 size;
    };
    
    struct RectWithEndpoint {
        vec2 p0;
        vec2 p1;
    };
    
    RectWithEndpoint to_rect_with_endpoint(RectWithSize rect) {
        RectWithEndpoint result;
        result.p0 = rect.p0;
        result.p1 = rect.p0 + rect.size;
    
        return result;
    }
    
    RectWithSize to_rect_with_size(RectWithEndpoint rect) {
        RectWithSize result;
        result.p0 = rect.p0;
        result.size = rect.p1 - rect.p0;
    
        return result;
    }
    
    RectWithSize intersect_rects(RectWithSize a, RectWithSize b) {
        RectWithSize result;
        result.p0 = max(a.p0, b.p0);
        result.size = min(a.p0 + a.size, b.p0 + b.size) - result.p0;
    
        return result;
    }
    
    float point_inside_rect(vec2 p, vec2 p0, vec2 p1) {
        vec2 s = step(p0, p) - step(p1, p);
        return s.x * s.y;
    }
    /* This Source Code Form is subject to the terms of the Mozilla Public
     * License, v. 2.0. If a copy of the MPL was not distributed with this
     * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
    
    
    #ifdef WR_VERTEX_SHADER
    #define VECS_PER_RENDER_TASK        2U
    
    uniform HIGHP_SAMPLER_FLOAT sampler2D sRenderTasks;
    
    struct RenderTaskCommonData {
        RectWithSize task_rect;
        float texture_layer_index;
    };
    
    struct RenderTaskData {
        RenderTaskCommonData common_data;
        vec3 user_data;
    };
    
    RenderTaskData fetch_render_task_data(int index) {
        ivec2 uv = get_fetch_uv(index, VECS_PER_RENDER_TASK);
    
        vec4 texel0 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(0, 0));
        vec4 texel1 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(1, 0));
    
        RectWithSize task_rect = RectWithSize(
            texel0.xy,
            texel0.zw
        );
    
        RenderTaskCommonData common_data = RenderTaskCommonData(
            task_rect,
            texel1.x
        );
    
        RenderTaskData data = RenderTaskData(
            common_data,
            texel1.yzw
        );
    
        return data;
    }
    
    RenderTaskCommonData fetch_render_task_common_data(int index) {
        ivec2 uv = get_fetch_uv(index, VECS_PER_RENDER_TASK);
    
        vec4 texel0 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(0, 0));
        vec4 texel1 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(1, 0));
    
        RectWithSize task_rect = RectWithSize(
            texel0.xy,
            texel0.zw
        );
    
        RenderTaskCommonData data = RenderTaskCommonData(
            task_rect,
            texel1.x
        );
    
        return data;
    }
    
    #define PIC_TYPE_IMAGE          1
    #define PIC_TYPE_TEXT_SHADOW    2
    
    /*
     The dynamic picture that this brush exists on. Right now, it
     contains minimal information. In the future, it will describe
     the transform mode of primitives on this picture, among other things.
     */
    struct PictureTask {
        RenderTaskCommonData common_data;
        float device_pixel_scale;
        vec2 content_origin;
    };
    
    PictureTask fetch_picture_task(int address) {
        RenderTaskData task_data = fetch_render_task_data(address);
    
        PictureTask task = PictureTask(
            task_data.common_data,
            task_data.user_data.x,
            task_data.user_data.yz
        );
    
        return task;
    }
    
    #define CLIP_TASK_EMPTY 0x7FFF
    
    struct ClipArea {
        RenderTaskCommonData common_data;
        float device_pixel_scale;
        vec2 screen_origin;
    };
    
    ClipArea fetch_clip_area(int index) {
        ClipArea area;
    
        if (index >= CLIP_TASK_EMPTY) {
            RectWithSize rect = RectWithSize(vec2(0.0), vec2(0.0));
    
            area.common_data = RenderTaskCommonData(rect, 0.0);
            area.device_pixel_scale = 0.0;
            area.screen_origin = vec2(0.0);
        } else {
            RenderTaskData task_data = fetch_render_task_data(index);
    
            area.common_data = task_data.common_data;
            area.device_pixel_scale = task_data.user_data.x;
            area.screen_origin = task_data.user_data.yz;
        }
    
        return area;
    }
    
    #endif //WR_VERTEX_SHADER
    /* This Source Code Form is subject to the terms of the Mozilla Public
     * License, v. 2.0. If a copy of the MPL was not distributed with this
     * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
    
    uniform HIGHP_SAMPLER_FLOAT sampler2D sGpuCache;
    
    #define VECS_PER_IMAGE_RESOURCE     2
    
    // TODO(gw): This is here temporarily while we have
    //           both GPU store and cache. When the GPU
    //           store code is removed, we can change the
    //           PrimitiveInstance instance structure to
    //           use 2x unsigned shorts as vertex attributes
    //           instead of an int, and encode the UV directly
    //           in the vertices.
    ivec2 get_gpu_cache_uv(HIGHP_FS_ADDRESS int address) {
        return ivec2(uint(address) % WR_MAX_VERTEX_TEXTURE_WIDTH,
                     uint(address) / WR_MAX_VERTEX_TEXTURE_WIDTH);
    }
    
    vec4[2] fetch_from_gpu_cache_2_direct(ivec2 address) {
        return vec4[2](
            TEXEL_FETCH(sGpuCache, address, 0, ivec2(0, 0)),
            TEXEL_FETCH(sGpuCache, address, 0, ivec2(1, 0))
        );
    }
    
    vec4[2] fetch_from_gpu_cache_2(HIGHP_FS_ADDRESS int address) {
        ivec2 uv = get_gpu_cache_uv(address);
        return vec4[2](
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(0, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(1, 0))
        );
    }
    
    vec4 fetch_from_gpu_cache_1_direct(ivec2 address) {
        return texelFetch(sGpuCache, address, 0);
    }
    
    vec4 fetch_from_gpu_cache_1(HIGHP_FS_ADDRESS int address) {
        ivec2 uv = get_gpu_cache_uv(address);
        return texelFetch(sGpuCache, uv, 0);
    }
    
    #ifdef WR_VERTEX_SHADER
    
    vec4[8] fetch_from_gpu_cache_8(int address) {
        ivec2 uv = get_gpu_cache_uv(address);
        return vec4[8](
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(0, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(1, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(2, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(3, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(4, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(5, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(6, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(7, 0))
        );
    }
    
    vec4[3] fetch_from_gpu_cache_3(int address) {
        ivec2 uv = get_gpu_cache_uv(address);
        return vec4[3](
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(0, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(1, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(2, 0))
        );
    }
    
    vec4[3] fetch_from_gpu_cache_3_direct(ivec2 address) {
        return vec4[3](
            TEXEL_FETCH(sGpuCache, address, 0, ivec2(0, 0)),
            TEXEL_FETCH(sGpuCache, address, 0, ivec2(1, 0)),
            TEXEL_FETCH(sGpuCache, address, 0, ivec2(2, 0))
        );
    }
    
    vec4[4] fetch_from_gpu_cache_4_direct(ivec2 address) {
        return vec4[4](
            TEXEL_FETCH(sGpuCache, address, 0, ivec2(0, 0)),
            TEXEL_FETCH(sGpuCache, address, 0, ivec2(1, 0)),
            TEXEL_FETCH(sGpuCache, address, 0, ivec2(2, 0)),
            TEXEL_FETCH(sGpuCache, address, 0, ivec2(3, 0))
        );
    }
    
    vec4[4] fetch_from_gpu_cache_4(int address) {
        ivec2 uv = get_gpu_cache_uv(address);
        return vec4[4](
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(0, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(1, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(2, 0)),
            TEXEL_FETCH(sGpuCache, uv, 0, ivec2(3, 0))
        );
    }
    
    //TODO: image resource is too specific for this module
    
    struct ImageResource {
        RectWithEndpoint uv_rect;
        float layer;
        vec3 user_data;
    };
    
    ImageResource fetch_image_resource(int address) {
        //Note: number of blocks has to match `renderer::BLOCKS_PER_UV_RECT`
        vec4 data[2] = fetch_from_gpu_cache_2(address);
        RectWithEndpoint uv_rect = RectWithEndpoint(data[0].xy, data[0].zw);
        return ImageResource(uv_rect, data[1].x, data[1].yzw);
    }
    
    ImageResource fetch_image_resource_direct(ivec2 address) {
        vec4 data[2] = fetch_from_gpu_cache_2_direct(address);
        RectWithEndpoint uv_rect = RectWithEndpoint(data[0].xy, data[0].zw);
        return ImageResource(uv_rect, data[1].x, data[1].yzw);
    }
    
    // Fetch optional extra data for a texture cache resource. This can contain
    // a polygon defining a UV rect within the texture cache resource.
    // Note: the polygon coordinates are in homogeneous space.
    struct ImageResourceExtra {
        vec4 st_tl;
        vec4 st_tr;
        vec4 st_bl;
        vec4 st_br;
    };
    
    ImageResourceExtra fetch_image_resource_extra(int address) {
        vec4 data[4] = fetch_from_gpu_cache_4(address + VECS_PER_IMAGE_RESOURCE);
        return ImageResourceExtra(
            data[0],
            data[1],
            data[2],
            data[3]
        );
    }
    
    #endif //WR_VERTEX_SHADER
    /* This Source Code Form is subject to the terms of the Mozilla Public
     * License, v. 2.0. If a copy of the MPL was not distributed with this
     * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
    
    flat varying vec4 vTransformBounds;
    
    #ifdef WR_VERTEX_SHADER
    
    #define VECS_PER_TRANSFORM   8U
    uniform HIGHP_SAMPLER_FLOAT sampler2D sTransformPalette;
    
    void init_transform_vs(vec4 local_bounds) {
        vTransformBounds = local_bounds;
    }
    
    struct Transform {
        mat4 m;
        mat4 inv_m;
        bool is_axis_aligned;
    };
    
    Transform fetch_transform(int id) {
        Transform transform;
    
        transform.is_axis_aligned = (id >> 24) == 0;
        int index = id & 0x00ffffff;
    
        // Create a UV base coord for each 8 texels.
        // This is required because trying to use an offset
        // of more than 8 texels doesn't work on some versions
        // of macOS.
        ivec2 uv = get_fetch_uv(index, VECS_PER_TRANSFORM);
        ivec2 uv0 = ivec2(uv.x + 0, uv.y);
    
        transform.m[0] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(0, 0));
        transform.m[1] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(1, 0));
        transform.m[2] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(2, 0));
        transform.m[3] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(3, 0));
    
        transform.inv_m[0] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(4, 0));
        transform.inv_m[1] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(5, 0));
        transform.inv_m[2] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(6, 0));
        transform.inv_m[3] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(7, 0));
    
        return transform;
    }
    
    // Return the intersection of the plane (set up by "normal" and "point")
    // with the ray (set up by "ray_origin" and "ray_dir"),
    // writing the resulting scaler into "t".
    bool ray_plane(vec3 normal, vec3 pt, vec3 ray_origin, vec3 ray_dir, out float t)
    {
        float denom = dot(normal, ray_dir);
        if (abs(denom) > 1e-6) {
            vec3 d = pt - ray_origin;
            t = dot(d, normal) / denom;
            return t >= 0.0;
        }
    
        return false;
    }
    
    // Apply the inverse transform "inv_transform"
    // to the reference point "ref" in CSS space,
    // producing a local point on a Transform plane,
    // set by a base point "a" and a normal "n".
    vec4 untransform(vec2 ref, vec3 n, vec3 a, mat4 inv_transform) {
        vec3 p = vec3(ref, -10000.0);
        vec3 d = vec3(0, 0, 1.0);
    
        float t = 0.0;
        // get an intersection of the Transform plane with Z axis vector,
        // originated from the "ref" point
        ray_plane(n, a, p, d, t);
        float z = p.z + d.z * t; // Z of the visible point on the Transform
    
        vec4 r = inv_transform * vec4(ref, z, 1.0);
        return r;
    }
    
    // Given a CSS space position, transform it back into the Transform space.
    vec4 get_node_pos(vec2 pos, Transform transform) {
        // get a point on the scroll node plane
        vec4 ah = transform.m * vec4(0.0, 0.0, 0.0, 1.0);
        vec3 a = ah.xyz / ah.w;
    
        // get the normal to the scroll node plane
        vec3 n = transpose(mat3(transform.inv_m)) * vec3(0.0, 0.0, 1.0);
        return untransform(pos, n, a, transform.inv_m);
    }
    
    #endif //WR_VERTEX_SHADER
    
    #ifdef WR_FRAGMENT_SHADER
    
    float signed_distance_rect(vec2 pos, vec2 p0, vec2 p1) {
        vec2 d = max(p0 - pos, pos - p1);
        return length(max(vec2(0.0), d)) + min(0.0, max(d.x, d.y));
    }
    
    float init_transform_fs(vec2 local_pos) {
        // Get signed distance from local rect bounds.
        float d = signed_distance_rect(
            local_pos,
            vTransformBounds.xy,
            vTransformBounds.zw
        );
    
        // Find the appropriate distance to apply the AA smoothstep over.
        float aa_range = compute_aa_range(local_pos);
    
        // Only apply AA to fragments outside the signed distance field.
        return distance_aa(aa_range, d);
    }
    
    float init_transform_rough_fs(vec2 local_pos) {
        return point_inside_rect(
            local_pos,
            vTransformBounds.xy,
            vTransformBounds.zw
        );
    }
    
    #endif //WR_FRAGMENT_SHADER
    
    #define EXTEND_MODE_CLAMP  0
    #define EXTEND_MODE_REPEAT 1
    
    #define SUBPX_DIR_NONE        0
    #define SUBPX_DIR_HORIZONTAL  1
    #define SUBPX_DIR_VERTICAL    2
    #define SUBPX_DIR_MIXED       3
    
    #define RASTER_LOCAL            0
    #define RASTER_SCREEN           1
    
    uniform sampler2DArray sPrevPassAlpha;
    uniform sampler2DArray sPrevPassColor;
    
    vec2 clamp_rect(vec2 pt, RectWithSize rect) {
        return clamp(pt, rect.p0, rect.p0 + rect.size);
    }
    
    // TODO: convert back to RectWithEndPoint if driver issues are resolved, if ever.
    flat varying vec4 vClipMaskUvBounds;
    // XY and W are homogeneous coordinates, Z is the layer index
    varying vec4 vClipMaskUv;
    
    
    #ifdef WR_VERTEX_SHADER
    
    #define COLOR_MODE_FROM_PASS          0
    #define COLOR_MODE_ALPHA              1
    #define COLOR_MODE_SUBPX_CONST_COLOR  2
    #define COLOR_MODE_SUBPX_BG_PASS0     3
    #define COLOR_MODE_SUBPX_BG_PASS1     4
    #define COLOR_MODE_SUBPX_BG_PASS2     5
    #define COLOR_MODE_SUBPX_DUAL_SOURCE  6
    #define COLOR_MODE_BITMAP             7
    #define COLOR_MODE_COLOR_BITMAP       8
    #define COLOR_MODE_IMAGE              9
    
    uniform HIGHP_SAMPLER_FLOAT sampler2D sPrimitiveHeadersF;
    uniform HIGHP_SAMPLER_FLOAT isampler2D sPrimitiveHeadersI;
    
    // Instanced attributes
    in ivec4 aData;
    
    #define VECS_PER_PRIM_HEADER_F 2U
    #define VECS_PER_PRIM_HEADER_I 2U
    
    struct PrimitiveHeader {
        RectWithSize local_rect;
        RectWithSize local_clip_rect;
        float z;
        int specific_prim_address;
        int transform_id;
        ivec4 user_data;
    };
    
    PrimitiveHeader fetch_prim_header(int index) {
        PrimitiveHeader ph;
    
        ivec2 uv_f = get_fetch_uv(index, VECS_PER_PRIM_HEADER_F);
        vec4 local_rect = TEXEL_FETCH(sPrimitiveHeadersF, uv_f, 0, ivec2(0, 0));
        vec4 local_clip_rect = TEXEL_FETCH(sPrimitiveHeadersF, uv_f, 0, ivec2(1, 0));
        ph.local_rect = RectWithSize(local_rect.xy, local_rect.zw);
        ph.local_clip_rect = RectWithSize(local_clip_rect.xy, local_clip_rect.zw);
    
        ivec2 uv_i = get_fetch_uv(index, VECS_PER_PRIM_HEADER_I);
        ivec4 data0 = TEXEL_FETCH(sPrimitiveHeadersI, uv_i, 0, ivec2(0, 0));
        ivec4 data1 = TEXEL_FETCH(sPrimitiveHeadersI, uv_i, 0, ivec2(1, 0));
        ph.z = float(data0.x);
        ph.specific_prim_address = data0.y;
        ph.transform_id = data0.z;
        ph.user_data = data1;
    
        return ph;
    }
    
    struct VertexInfo {
        vec2 local_pos;
        vec2 snap_offset;
        vec4 world_pos;
    };
    
    VertexInfo write_vertex(RectWithSize instance_rect,
                            RectWithSize local_clip_rect,
                            float z,
                            Transform transform,
                            PictureTask task) {
    
        // Select the corner of the local rect that we are processing.
        vec2 local_pos = instance_rect.p0 + instance_rect.size * aPosition.xy;
    
        // Clamp to the two local clip rects.
        vec2 clamped_local_pos = clamp_rect(local_pos, local_clip_rect);
    
        // Transform the current vertex to world space.
        vec4 world_pos = transform.m * vec4(clamped_local_pos, 0.0, 1.0);
    
        // Convert the world positions to device pixel space.
        vec2 device_pos = world_pos.xy * task.device_pixel_scale;
    
        // Apply offsets for the render task to get correct screen location.
        vec2 final_offset = -task.content_origin + task.common_data.task_rect.p0;
    
        gl_Position = uTransform * vec4(device_pos + final_offset * world_pos.w, z * world_pos.w, world_pos.w);
    
        VertexInfo vi = VertexInfo(
            clamped_local_pos,
            vec2(0.0, 0.0),
            world_pos
        );
    
        return vi;
    }
    
    float cross2(vec2 v0, vec2 v1) {
        return v0.x * v1.y - v0.y * v1.x;
    }
    
    // Return intersection of line (p0,p1) and line (p2,p3)
    vec2 intersect_lines(vec2 p0, vec2 p1, vec2 p2, vec2 p3) {
        vec2 d0 = p0 - p1;
        vec2 d1 = p2 - p3;
    
        float s0 = cross2(p0, p1);
        float s1 = cross2(p2, p3);
    
        float d = cross2(d0, d1);
        float nx = s0 * d1.x - d0.x * s1;
        float ny = s0 * d1.y - d0.y * s1;
    
        return vec2(nx / d, ny / d);
    }
    
    VertexInfo write_transform_vertex(RectWithSize local_segment_rect,
                                      RectWithSize local_prim_rect,
                                      RectWithSize local_clip_rect,
                                      vec4 clip_edge_mask,
                                      float z,
                                      Transform transform,
                                      PictureTask task) {
        // Calculate a clip rect from local_rect + local clip
        RectWithEndpoint clip_rect = to_rect_with_endpoint(local_clip_rect);
        RectWithEndpoint segment_rect = to_rect_with_endpoint(local_segment_rect);
        segment_rect.p0 = clamp(segment_rect.p0, clip_rect.p0, clip_rect.p1);
        segment_rect.p1 = clamp(segment_rect.p1, clip_rect.p0, clip_rect.p1);
    
        // Calculate a clip rect from local_rect + local clip
        RectWithEndpoint prim_rect = to_rect_with_endpoint(local_prim_rect);
        prim_rect.p0 = clamp(prim_rect.p0, clip_rect.p0, clip_rect.p1);
        prim_rect.p1 = clamp(prim_rect.p1, clip_rect.p0, clip_rect.p1);
    
        // As this is a transform shader, extrude by 2 (local space) pixels
        // in each direction. This gives enough space around the edge to
        // apply distance anti-aliasing. Technically, it:
        // (a) slightly over-estimates the number of required pixels in the simple case.
        // (b) might not provide enough edge in edge case perspective projections.
        // However, it's fast and simple. If / when we ever run into issues, we
        // can do some math on the projection matrix to work out a variable
        // amount to extrude.
    
        // Only extrude along edges where we are going to apply AA.
        float extrude_amount = 2.0;
        vec4 extrude_distance = vec4(extrude_amount) * clip_edge_mask;
        local_segment_rect.p0 -= extrude_distance.xy;
        local_segment_rect.size += extrude_distance.xy + extrude_distance.zw;
    
        // Select the corner of the local rect that we are processing.
        vec2 local_pos = local_segment_rect.p0 + local_segment_rect.size * aPosition.xy;
    
        // Convert the world positions to device pixel space.
        vec2 task_offset = task.common_data.task_rect.p0 - task.content_origin;
    
        // Transform the current vertex to the world cpace.
        vec4 world_pos = transform.m * vec4(local_pos, 0.0, 1.0);
        vec4 final_pos = vec4(
            world_pos.xy * task.device_pixel_scale + task_offset * world_pos.w,
            z * world_pos.w,
            world_pos.w
        );
    
        gl_Position = uTransform * final_pos;
    
        init_transform_vs(mix(
            vec4(prim_rect.p0, prim_rect.p1),
            vec4(segment_rect.p0, segment_rect.p1),
            clip_edge_mask
        ));
    
        VertexInfo vi = VertexInfo(
            local_pos,
            vec2(0.0),
            world_pos
        );
    
        return vi;
    }
    
    void write_clip(vec4 world_pos, vec2 snap_offset, ClipArea area) {
        vec2 uv = world_pos.xy * area.device_pixel_scale +
            world_pos.w * (snap_offset + area.common_data.task_rect.p0 - area.screen_origin);
        vClipMaskUvBounds = vec4(
            area.common_data.task_rect.p0,
            area.common_data.task_rect.p0 + area.common_data.task_rect.size
        );
        vClipMaskUv = vec4(uv, area.common_data.texture_layer_index, world_pos.w);
    }
    
    // Read the exta image data containing the homogeneous screen space coordinates
    // of the corners, interpolate between them, and return real screen space UV.
    vec2 get_image_quad_uv(int address, vec2 f) {
        ImageResourceExtra extra_data = fetch_image_resource_extra(address);
        vec4 x = mix(extra_data.st_tl, extra_data.st_tr, f.x);
        vec4 y = mix(extra_data.st_bl, extra_data.st_br, f.x);
        vec4 z = mix(x, y, f.y);
        return z.xy / z.w;
    }
    #endif //WR_VERTEX_SHADER
    
    #ifdef WR_FRAGMENT_SHADER
    
    float do_clip() {
        // check for the dummy bounds, which are given to the opaque objects
        if (vClipMaskUvBounds.xy == vClipMaskUvBounds.zw) {
            return 1.0;
        }
        // anything outside of the mask is considered transparent
        //Note: we assume gl_FragCoord.w == interpolated(1 / vClipMaskUv.w)
        vec2 mask_uv = vClipMaskUv.xy * gl_FragCoord.w;
        bvec2 left = lessThanEqual(vClipMaskUvBounds.xy, mask_uv); // inclusive
        bvec2 right = greaterThan(vClipMaskUvBounds.zw, mask_uv); // non-inclusive
        // bail out if the pixel is outside the valid bounds
        if (!all(bvec4(left, right))) {
            return 0.0;
        }
        // finally, the slow path - fetch the mask value from an image
        // Note the Z getting rounded to the nearest integer because the variable
        // is still interpolated and becomes a subject of precision-caused
        // fluctuations, see https://bugzilla.mozilla.org/show_bug.cgi?id=1491911
        ivec3 tc = ivec3(mask_uv, vClipMaskUv.z + 0.5);
        return texelFetch(sPrevPassAlpha, tc, 0).r;
    }
    
    #ifdef WR_FEATURE_DITHERING
    vec4 dither(vec4 color) {
        const int matrix_mask = 7;
    
        ivec2 pos = ivec2(gl_FragCoord.xy) & ivec2(matrix_mask);
        float noise_normalized = (texelFetch(sDither, pos, 0).r * 255.0 + 0.5) / 64.0;
        float noise = (noise_normalized - 0.5) / 256.0; // scale down to the unit length
    
        return color + vec4(noise, noise, noise, 0);
    }
    #else
    vec4 dither(vec4 color) {
        return color;
    }
    #endif //WR_FEATURE_DITHERING
    
    vec4 sample_gradient(HIGHP_FS_ADDRESS int address, float offset, float gradient_repeat) {
        // Modulo the offset if the gradient repeats.
        float x = mix(offset, fract(offset), gradient_repeat);
    
        // Calculate the color entry index to use for this offset:
        //     offsets < 0 use the first color entry, 0
        //     offsets from [0, 1) use the color entries in the range of [1, N-1)
        //     offsets >= 1 use the last color entry, N-1
        //     so transform the range [0, 1) -> [1, N-1)
    
        // TODO(gw): In the future we might consider making the size of the
        // LUT vary based on number / distribution of stops in the gradient.
        const int GRADIENT_ENTRIES = 128;
        x = 1.0 + x * float(GRADIENT_ENTRIES);
    
        // Calculate the texel to index into the gradient color entries:
        //     floor(x) is the gradient color entry index
        //     fract(x) is the linear filtering factor between start and end
        int lut_offset = 2 * int(floor(x));     // There is a [start, end] color per entry.
    
        // Ensure we don't fetch outside the valid range of the LUT.
        lut_offset = clamp(lut_offset, 0, 2 * (GRADIENT_ENTRIES + 1));
    
        // Fetch the start and end color.
        vec4 texels[2] = fetch_from_gpu_cache_2(address + lut_offset);
    
        // Finally interpolate and apply dithering
        return dither(mix(texels[0], texels[1], fract(x)));
    }
    
    #endif //WR_FRAGMENT_SHADER
    
    flat varying vec4 vColor;
    varying vec3 vUv;
    flat varying vec4 vUvBorder;
    flat varying vec2 vMaskSwizzle;
    
    #ifdef WR_FEATURE_GLYPH_TRANSFORM
    varying vec4 vUvClip;
    #endif
    
    #ifdef WR_VERTEX_SHADER
    
    #define VECS_PER_TEXT_RUN           2
    #define GLYPHS_PER_GPU_BLOCK        2U
    
    #ifdef WR_FEATURE_GLYPH_TRANSFORM
    RectWithSize transform_rect(RectWithSize rect, mat2 transform) {
        vec2 center = transform * (rect.p0 + rect.size * 0.5);
        vec2 radius = mat2(abs(transform[0]), abs(transform[1])) * (rect.size * 0.5);
        return RectWithSize(center - radius, radius * 2.0);
    }
    
    bool rect_inside_rect(RectWithSize little, RectWithSize big) {
        return all(lessThanEqual(vec4(big.p0, little.p0 + little.size),
                                 vec4(little.p0, big.p0 + big.size)));
    }
    #endif //WR_FEATURE_GLYPH_TRANSFORM
    
    struct Glyph {
        vec2 offset;
    };
    
    Glyph fetch_glyph(int specific_prim_address,
                      int glyph_index) {
        // Two glyphs are packed in each texel in the GPU cache.
        int glyph_address = specific_prim_address +
                            VECS_PER_TEXT_RUN +
                            int(uint(glyph_index) / GLYPHS_PER_GPU_BLOCK);
        vec4 data = fetch_from_gpu_cache_1(glyph_address);
        // Select XY or ZW based on glyph index.
        // We use "!= 0" instead of "== 1" here in order to work around a driver
        // bug with equality comparisons on integers.
        vec2 glyph = mix(data.xy, data.zw,
                         bvec2(uint(glyph_index) % GLYPHS_PER_GPU_BLOCK != 0U));
    
        return Glyph(glyph);
    }
    
    struct GlyphResource {
        vec4 uv_rect;
        float layer;
        vec2 offset;
        float scale;
    };
    
    GlyphResource fetch_glyph_resource(int address) {
        vec4 data[2] = fetch_from_gpu_cache_2(address);
        return GlyphResource(data[0], data[1].x, data[1].yz, data[1].w);
    }
    
    struct TextRun {
        vec4 color;
        vec4 bg_color;
    };
    
    TextRun fetch_text_run(int address) {
        vec4 data[2] = fetch_from_gpu_cache_2(address);
        return TextRun(data[0], data[1]);
    }
    
    VertexInfo write_text_vertex(RectWithSize local_clip_rect,
                                 float z,
                                 int raster_space,
                                 Transform transform,
                                 PictureTask task,
                                 vec2 text_offset,
                                 vec2 glyph_offset,
                                 RectWithSize glyph_rect,
                                 vec2 snap_bias) {
        // The offset to snap the glyph rect to a device pixel
        vec2 snap_offset = vec2(0.0);
        // Transform from glyph space to local space
        mat2 glyph_transform_inv = mat2(1.0);
    
    #ifdef WR_FEATURE_GLYPH_TRANSFORM
        bool remove_subpx_offset = true;
    #else
        bool remove_subpx_offset = transform.is_axis_aligned;
    #endif
        // Compute the snapping offset only if the scroll node transform is axis-aligned.
        if (remove_subpx_offset) {
            // Be careful to only snap with the transform when in screen raster space.
            switch (raster_space) {
                case RASTER_SCREEN: {
                    // Transform from local space to glyph space.
                    float device_scale = task.device_pixel_scale / transform.m[3].w;
                    mat2 glyph_transform = mat2(transform.m) * device_scale;
    
                    // Ensure the transformed text offset does not contain a subpixel translation
                    // such that glyph snapping is stable for equivalent glyph subpixel positions.
                    vec2 device_text_pos = glyph_transform * text_offset + transform.m[3].xy * device_scale;
                    snap_offset = floor(device_text_pos + 0.5) - device_text_pos;
    
                    // Snap the glyph offset to a device pixel, using an appropriate bias depending
                    // on whether subpixel positioning is required.
                    vec2 device_glyph_offset = glyph_transform * glyph_offset;
                    snap_offset += floor(device_glyph_offset + snap_bias) - device_glyph_offset;
    
                    // Transform from glyph space back to local space.
                    glyph_transform_inv = inverse(glyph_transform);
    
    #ifndef WR_FEATURE_GLYPH_TRANSFORM
                    // If not using transformed subpixels, the glyph rect is actually in local space.
                    // So convert the snap offset back to local space.
                    snap_offset = glyph_transform_inv * snap_offset;
    #endif
                    break;
                }
                default: {
                    // Otherwise, when in local raster space, the transform may be animated, so avoid
                    // snapping with the transform to avoid oscillation.
                    snap_offset = floor(text_offset + 0.5) - text_offset;
                    snap_offset += floor(glyph_offset + snap_bias) - glyph_offset;
                    break;
                }
            }
        }
    
        // Actually translate the glyph rect to a device pixel using the snap offset.
        glyph_rect.p0 += snap_offset;
    
    #ifdef WR_FEATURE_GLYPH_TRANSFORM
        // The glyph rect is in device space, so transform it back to local space.
        RectWithSize local_rect = transform_rect(glyph_rect, glyph_transform_inv);
    
        // Select the corner of the glyph's local space rect that we are processing.
        vec2 local_pos = local_rect.p0 + local_rect.size * aPosition.xy;
    
        // If the glyph's local rect would fit inside the local clip rect, then select a corner from
        // the device space glyph rect to reduce overdraw of clipped pixels in the fragment shader.
        // Otherwise, fall back to clamping the glyph's local rect to the local clip rect.
        if (rect_inside_rect(local_rect, local_clip_rect)) {
            local_pos = glyph_transform_inv * (glyph_rect.p0 + glyph_rect.size * aPosition.xy);
        }
    #else
        // Select the corner of the glyph rect that we are processing.
        vec2 local_pos = glyph_rect.p0 + glyph_rect.size * aPosition.xy;
    #endif
    
        // Clamp to the local clip rect.
        local_pos = clamp_rect(local_pos, local_clip_rect);
    
        // Map the clamped local space corner into device space.
        vec4 world_pos = transform.m * vec4(local_pos, 0.0, 1.0);
        vec2 device_pos = world_pos.xy * task.device_pixel_scale;
    
        // Apply offsets for the render task to get correct screen location.
        vec2 final_offset = -task.content_origin + task.common_data.task_rect.p0;
    
        gl_Position = uTransform * vec4(device_pos + final_offset * world_pos.w, z * world_pos.w, world_pos.w);
    
        VertexInfo vi = VertexInfo(
            local_pos,
            snap_offset,
            world_pos
        );
    
        return vi;
    }
    
    void main(void) {
        int prim_header_address = aData.x;
        int glyph_index = aData.y & 0xffff;
        int render_task_index = aData.y >> 16;
        int resource_address = aData.z;
        int raster_space = aData.w >> 16;
        int subpx_dir = (aData.w >> 8) & 0xff;
        int color_mode = aData.w & 0xff;
    
        PrimitiveHeader ph = fetch_prim_header(prim_header_address);
        Transform transform = fetch_transform(ph.transform_id);
        ClipArea clip_area = fetch_clip_area(ph.user_data.w);
        PictureTask task = fetch_picture_task(render_task_index);
    
        TextRun text = fetch_text_run(ph.specific_prim_address);
        vec2 text_offset = vec2(ph.user_data.xy) / 256.0;
    
        if (color_mode == COLOR_MODE_FROM_PASS) {
            color_mode = uMode;
        }
    
        Glyph glyph = fetch_glyph(ph.specific_prim_address, glyph_index);
        glyph.offset += ph.local_rect.p0 - text_offset;
    
        GlyphResource res = fetch_glyph_resource(resource_address);
    
    #ifdef WR_FEATURE_GLYPH_TRANSFORM
        // Transform from local space to glyph space.
        mat2 glyph_transform = mat2(transform.m) * task.device_pixel_scale;
    
        // Compute the glyph rect in glyph space.
        RectWithSize glyph_rect = RectWithSize(res.offset + glyph_transform * (text_offset + glyph.offset),
                                               res.uv_rect.zw - res.uv_rect.xy);
    #else
        float raster_scale = float(ph.user_data.z) / 65535.0;
    
        // Scale from glyph space to local space.
        float scale = res.scale / (raster_scale * task.device_pixel_scale);
    
        // Compute the glyph rect in local space.
        RectWithSize glyph_rect = RectWithSize(scale * res.offset + text_offset + glyph.offset,
                                               scale * (res.uv_rect.zw - res.uv_rect.xy));
    #endif
    
        vec2 snap_bias;
        // In subpixel mode, the subpixel offset has already been
        // accounted for while rasterizing the glyph. However, we
        // must still round with a subpixel bias rather than rounding
        // to the nearest whole pixel, depending on subpixel direciton.
        switch (subpx_dir) {
            case SUBPX_DIR_NONE:
            default:
                snap_bias = vec2(0.5);
                break;
            case SUBPX_DIR_HORIZONTAL:
                // Glyphs positioned [-0.125, 0.125] get a
                // subpx position of zero. So include that
                // offset in the glyph position to ensure
                // we round to the correct whole position.
                snap_bias = vec2(0.125, 0.5);
                break;
            case SUBPX_DIR_VERTICAL:
                snap_bias = vec2(0.5, 0.125);
                break;
            case SUBPX_DIR_MIXED:
                snap_bias = vec2(0.125);
                break;
        }
    
        VertexInfo vi = write_text_vertex(ph.local_clip_rect,
                                          ph.z,
                                          raster_space,
                                          transform,
                                          task,
                                          text_offset,
                                          glyph.offset,
                                          glyph_rect,
                                          snap_bias);
        glyph_rect.p0 += vi.snap_offset;
    
    #ifdef WR_FEATURE_GLYPH_TRANSFORM
        vec2 f = (glyph_transform * vi.local_pos - glyph_rect.p0) / glyph_rect.size;
        vUvClip = vec4(f, 1.0 - f);
    #else
        vec2 f = (vi.local_pos - glyph_rect.p0) / glyph_rect.size;
    #endif
    
        write_clip(vi.world_pos, vi.snap_offset, clip_area);
    
        switch (color_mode) {
            case COLOR_MODE_ALPHA:
            case COLOR_MODE_BITMAP:
                vMaskSwizzle = vec2(0.0, 1.0);
                vColor = text.color;
                break;
            case COLOR_MODE_SUBPX_BG_PASS2:
            case COLOR_MODE_SUBPX_DUAL_SOURCE:
                vMaskSwizzle = vec2(1.0, 0.0);
                vColor = text.color;
                break;
            case COLOR_MODE_SUBPX_CONST_COLOR:
            case COLOR_MODE_SUBPX_BG_PASS0:
            case COLOR_MODE_COLOR_BITMAP:
                vMaskSwizzle = vec2(1.0, 0.0);
                vColor = vec4(text.color.a);
                break;
            case COLOR_MODE_SUBPX_BG_PASS1:
                vMaskSwizzle = vec2(-1.0, 1.0);
                vColor = vec4(text.color.a) * text.bg_color;
                break;
            default:
                vMaskSwizzle = vec2(0.0);
                vColor = vec4(1.0);
        }
    
        vec2 texture_size = vec2(textureSize(sColor0, 0));
        vec2 st0 = res.uv_rect.xy / texture_size;
        vec2 st1 = res.uv_rect.zw / texture_size;
    
        vUv = vec3(mix(st0, st1, f), res.layer);
        vUvBorder = (res.uv_rect + vec4(0.5, 0.5, -0.5, -0.5)) / texture_size.xyxy;
    }
    #endif
    
    #ifdef WR_FRAGMENT_SHADER
    void main(void) {
        vec3 tc = vec3(clamp(vUv.xy, vUvBorder.xy, vUvBorder.zw), vUv.z);
        vec4 mask = texture(sColor0, tc);
        mask.rgb = mask.rgb * vMaskSwizzle.x + mask.aaa * vMaskSwizzle.y;
    
        float alpha = do_clip();
    #ifdef WR_FEATURE_GLYPH_TRANSFORM
        alpha *= float(all(greaterThanEqual(vUvClip, vec4(0.0))));
    #endif
    
    #if defined(WR_FEATURE_DEBUG_OVERDRAW)
        oFragColor = WR_DEBUG_OVERDRAW_COLOR;
    #elif defined(WR_FEATURE_DUAL_SOURCE_BLENDING)
        vec4 alpha_mask = mask * alpha;
        oFragColor = vColor * alpha_mask;
        oFragBlend = alpha_mask * vColor.a;
    #else
        write_output(vColor * mask * alpha);
    #endif
    }
    #endif
    "#;

    let fs_source = br#"
    #version 150
// ps_text_run
#define WR_FRAGMENT_SHADER
#define WR_MAX_VERTEX_TEXTURE_WIDTH 1024U
#define WR_FEATURE_
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_FEATURE_PIXEL_LOCAL_STORAGE
// For now, we need both extensions here, in order to initialize
// the PLS to the current framebuffer color. In future, we can
// possibly remove that requirement, or at least support the
// other framebuffer fetch extensions that provide the same
// functionality.
#extension GL_EXT_shader_pixel_local_storage : require
#extension GL_ARM_shader_framebuffer_fetch : require
#endif

#ifdef WR_FEATURE_TEXTURE_EXTERNAL
// Please check https://www.khronos.org/registry/OpenGL/extensions/OES/OES_EGL_image_external_essl3.txt
// for this extension.
#extension GL_OES_EGL_image_external_essl3 : require
#endif

#ifdef WR_FEATURE_ADVANCED_BLEND
#extension GL_KHR_blend_equation_advanced : require
#endif

#ifdef WR_FEATURE_DUAL_SOURCE_BLENDING
#ifdef GL_ES
#extension GL_EXT_blend_func_extended : require
#else
#extension GL_ARB_explicit_attrib_location : require
#endif
#endif

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#if defined(GL_ES)
    #if GL_ES == 1
        #ifdef GL_FRAGMENT_PRECISION_HIGH
        precision highp sampler2DArray;
        #else
        precision mediump sampler2DArray;
        #endif

        // Sampler default precision is lowp on mobile GPUs.
        // This causes RGBA32F texture data to be clamped to 16 bit floats on some GPUs (e.g. Mali-T880).
        // Define highp precision macro to allow lossless FLOAT texture sampling.
        #define HIGHP_SAMPLER_FLOAT highp

        // Default int precision in GLES 3 is highp (32 bits) in vertex shaders
        // and mediump (16 bits) in fragment shaders. If an int is being used as
        // a texel address in a fragment shader it, and therefore requires > 16
        // bits, it must be qualified with this.
        #define HIGHP_FS_ADDRESS highp

        // texelFetchOffset is buggy on some Android GPUs (see issue #1694).
        // Fallback to texelFetch on mobile GPUs.
        #define TEXEL_FETCH(sampler, position, lod, offset) texelFetch(sampler, position + offset, lod)
    #else
        #define HIGHP_SAMPLER_FLOAT
        #define HIGHP_FS_ADDRESS
        #define TEXEL_FETCH(sampler, position, lod, offset) texelFetchOffset(sampler, position, lod, offset)
    #endif
#else
    #define HIGHP_SAMPLER_FLOAT
    #define HIGHP_FS_ADDRESS
    #define TEXEL_FETCH(sampler, position, lod, offset) texelFetchOffset(sampler, position, lod, offset)
#endif

#ifdef WR_VERTEX_SHADER
    #define varying out
#endif

#ifdef WR_FRAGMENT_SHADER
    precision highp float;
    #define varying in
#endif

#if defined(WR_FEATURE_TEXTURE_EXTERNAL) || defined(WR_FEATURE_TEXTURE_RECT) || defined(WR_FEATURE_TEXTURE_2D)
#define TEX_SAMPLE(sampler, tex_coord) texture(sampler, tex_coord.xy)
#else
#define TEX_SAMPLE(sampler, tex_coord) texture(sampler, tex_coord)
#endif

//======================================================================================
// Vertex shader attributes and uniforms
//======================================================================================
#ifdef WR_VERTEX_SHADER
    // A generic uniform that shaders can optionally use to configure
    // an operation mode for this batch.
    uniform int uMode;

    // Uniform inputs
    uniform mat4 uTransform;       // Orthographic projection

    // Attribute inputs
    in vec3 aPosition;

    // get_fetch_uv is a macro to work around a macOS Intel driver parsing bug.
    // TODO: convert back to a function once the driver issues are resolved, if ever.
    // https://github.com/servo/webrender/pull/623
    // https://github.com/servo/servo/issues/13953
    // Do the division with unsigned ints because that's more efficient with D3D
    #define get_fetch_uv(i, vpi)  ivec2(int(vpi * (uint(i) % (WR_MAX_VERTEX_TEXTURE_WIDTH/vpi))), int(uint(i) / (WR_MAX_VERTEX_TEXTURE_WIDTH/vpi)))
#endif

//======================================================================================
// Fragment shader attributes and uniforms
//======================================================================================
#ifdef WR_FRAGMENT_SHADER
    // Uniform inputs

    #ifdef WR_FEATURE_PIXEL_LOCAL_STORAGE
        // Define the storage class of the pixel local storage.
        // If defined as writable, it's a compile time error to
        // have a normal fragment output variable declared.
        #if defined(PLS_READONLY)
            #define PLS_BLOCK __pixel_local_inEXT
        #elif defined(PLS_WRITEONLY)
            #define PLS_BLOCK __pixel_local_outEXT
        #else
            #define PLS_BLOCK __pixel_localEXT
        #endif

        // The structure of pixel local storage. Right now, it's
        // just the current framebuffer color. In future, we have
        // (at least) 12 bytes of space we can store extra info
        // here (such as clip mask values).
        PLS_BLOCK FrameBuffer {
            layout(rgba8) highp vec4 color;
        } PLS;

        #ifndef PLS_READONLY
        // Write the output of a fragment shader to PLS. Applies
        // premultipled alpha blending by default, since the blender
        // is disabled when PLS is active.
        // TODO(gw): Properly support alpha blend mode for webgl / canvas.
        void write_output(vec4 color) {
            PLS.color = color + PLS.color * (1.0 - color.a);
        }

        // Write a raw value straight to PLS, if the fragment shader has
        // already applied blending.
        void write_output_raw(vec4 color) {
            PLS.color = color;
        }
        #endif

        #ifndef PLS_WRITEONLY
        // Retrieve the current framebuffer color. Useful in conjunction with
        // the write_output_raw function.
        vec4 get_current_framebuffer_color() {
            return PLS.color;
        }
        #endif
    #else
        // Fragment shader outputs
        #ifdef WR_FEATURE_ADVANCED_BLEND
            layout(blend_support_all_equations) out;
        #endif

        #ifdef WR_FEATURE_DUAL_SOURCE_BLENDING
            layout(location = 0, index = 0) out vec4 oFragColor;
            layout(location = 0, index = 1) out vec4 oFragBlend;
        #else
            out vec4 oFragColor;
        #endif

        // Write an output color in normal (non-PLS) shaders.
        void write_output(vec4 color) {
            oFragColor = color;
        }
    #endif

    #define EPSILON                     0.0001

    // "Show Overdraw" color. Premultiplied.
    #define WR_DEBUG_OVERDRAW_COLOR     vec4(0.110, 0.077, 0.027, 0.125)

    float distance_to_line(vec2 p0, vec2 perp_dir, vec2 p) {
        vec2 dir_to_p0 = p0 - p;
        return dot(normalize(perp_dir), dir_to_p0);
    }

    /// Find the appropriate half range to apply the AA approximation over.
    /// This range represents a coefficient to go from one CSS pixel to half a device pixel.
    float compute_aa_range(vec2 position) {
        // The constant factor is chosen to compensate for the fact that length(fw) is equal
        // to sqrt(2) times the device pixel ratio in the typical case. 0.5/sqrt(2) = 0.35355.
        //
        // This coefficient is chosen to ensure that any sample 0.5 pixels or more inside of
        // the shape has no anti-aliasing applied to it (since pixels are sampled at their center,
        // such a pixel (axis aligned) is fully inside the border). We need this so that antialiased
        // curves properly connect with non-antialiased vertical or horizontal lines, among other things.
        //
        // Lines over a half-pixel away from the pixel center *can* intersect with the pixel square;
        // indeed, unless they are horizontal or vertical, they are guaranteed to. However, choosing
        // a nonzero area for such pixels causes noticeable artifacts at the junction between an anti-
        // aliased corner and a straight edge.
        //
        // We may want to adjust this constant in specific scenarios (for example keep the principled
        // value for straight edges where we want pixel-perfect equivalence with non antialiased lines
        // when axis aligned, while selecting a larger and smoother aa range on curves).
        return 0.35355 * length(fwidth(position));
    }

    /// Return the blending coefficient for distance antialiasing.
    ///
    /// 0.0 means inside the shape, 1.0 means outside.
    ///
    /// This cubic polynomial approximates the area of a 1x1 pixel square under a
    /// line, given the signed Euclidean distance from the center of the square to
    /// that line. Calculating the *exact* area would require taking into account
    /// not only this distance but also the angle of the line. However, in
    /// practice, this complexity is not required, as the area is roughly the same
    /// regardless of the angle.
    ///
    /// The coefficients of this polynomial were determined through least-squares
    /// regression and are accurate to within 2.16% of the total area of the pixel
    /// square 95% of the time, with a maximum error of 3.53%.
    ///
    /// See the comments in `compute_aa_range()` for more information on the
    /// cutoff values of -0.5 and 0.5.
    float distance_aa(float aa_range, float signed_distance) {
        float dist = 0.5 * signed_distance / aa_range;
        if (dist <= -0.5 + EPSILON)
            return 1.0;
        if (dist >= 0.5 - EPSILON)
            return 0.0;
        return 0.5 + dist * (0.8431027 * dist * dist - 1.14453603);
    }

    /// Component-wise selection.
    ///
    /// The idea of using this is to ensure both potential branches are executed before
    /// selecting the result, to avoid observable timing differences based on the condition.
    ///
    /// Example usage: color = if_then_else(LessThanEqual(color, vec3(0.5)), vec3(0.0), vec3(1.0));
    ///
    /// The above example sets each component to 0.0 or 1.0 independently depending on whether
    /// their values are below or above 0.5.
    ///
    /// This is written as a macro in order to work with vectors of any dimension.
    ///
    /// Note: Some older android devices don't support mix with bvec. If we ever run into them
    /// the only option we have is to polyfill it with a branch per component.
    #define if_then_else(cond, then_branch, else_branch) mix(else_branch, then_branch, cond)
#endif

//======================================================================================
// Shared shader uniforms
//======================================================================================
#ifdef WR_FEATURE_TEXTURE_2D
uniform sampler2D sColor0;
uniform sampler2D sColor1;
uniform sampler2D sColor2;
#elif defined WR_FEATURE_TEXTURE_RECT
uniform sampler2DRect sColor0;
uniform sampler2DRect sColor1;
uniform sampler2DRect sColor2;
#elif defined WR_FEATURE_TEXTURE_EXTERNAL
uniform samplerExternalOES sColor0;
uniform samplerExternalOES sColor1;
uniform samplerExternalOES sColor2;
#else
uniform sampler2DArray sColor0;
uniform sampler2DArray sColor1;
uniform sampler2DArray sColor2;
#endif

#ifdef WR_FEATURE_DITHERING
uniform sampler2D sDither;
#endif

//======================================================================================
// Interpolator definitions
//======================================================================================

//======================================================================================
// VS only types and UBOs
//======================================================================================

//======================================================================================
// VS only functions
//======================================================================================
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct RectWithSize {
    vec2 p0;
    vec2 size;
};

struct RectWithEndpoint {
    vec2 p0;
    vec2 p1;
};

RectWithEndpoint to_rect_with_endpoint(RectWithSize rect) {
    RectWithEndpoint result;
    result.p0 = rect.p0;
    result.p1 = rect.p0 + rect.size;

    return result;
}

RectWithSize to_rect_with_size(RectWithEndpoint rect) {
    RectWithSize result;
    result.p0 = rect.p0;
    result.size = rect.p1 - rect.p0;

    return result;
}

RectWithSize intersect_rects(RectWithSize a, RectWithSize b) {
    RectWithSize result;
    result.p0 = max(a.p0, b.p0);
    result.size = min(a.p0 + a.size, b.p0 + b.size) - result.p0;

    return result;
}

float point_inside_rect(vec2 p, vec2 p0, vec2 p1) {
    vec2 s = step(p0, p) - step(p1, p);
    return s.x * s.y;
}
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifdef WR_VERTEX_SHADER
#define VECS_PER_RENDER_TASK        2U

uniform HIGHP_SAMPLER_FLOAT sampler2D sRenderTasks;

struct RenderTaskCommonData {
    RectWithSize task_rect;
    float texture_layer_index;
};

struct RenderTaskData {
    RenderTaskCommonData common_data;
    vec3 user_data;
};

RenderTaskData fetch_render_task_data(int index) {
    ivec2 uv = get_fetch_uv(index, VECS_PER_RENDER_TASK);

    vec4 texel0 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(0, 0));
    vec4 texel1 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(1, 0));

    RectWithSize task_rect = RectWithSize(
        texel0.xy,
        texel0.zw
    );

    RenderTaskCommonData common_data = RenderTaskCommonData(
        task_rect,
        texel1.x
    );

    RenderTaskData data = RenderTaskData(
        common_data,
        texel1.yzw
    );

    return data;
}

RenderTaskCommonData fetch_render_task_common_data(int index) {
    ivec2 uv = get_fetch_uv(index, VECS_PER_RENDER_TASK);

    vec4 texel0 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(0, 0));
    vec4 texel1 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(1, 0));

    RectWithSize task_rect = RectWithSize(
        texel0.xy,
        texel0.zw
    );

    RenderTaskCommonData data = RenderTaskCommonData(
        task_rect,
        texel1.x
    );

    return data;
}

#define PIC_TYPE_IMAGE          1
#define PIC_TYPE_TEXT_SHADOW    2

/*
 The dynamic picture that this brush exists on. Right now, it
 contains minimal information. In the future, it will describe
 the transform mode of primitives on this picture, among other things.
 */
struct PictureTask {
    RenderTaskCommonData common_data;
    float device_pixel_scale;
    vec2 content_origin;
};

PictureTask fetch_picture_task(int address) {
    RenderTaskData task_data = fetch_render_task_data(address);

    PictureTask task = PictureTask(
        task_data.common_data,
        task_data.user_data.x,
        task_data.user_data.yz
    );

    return task;
}

#define CLIP_TASK_EMPTY 0x7FFF

struct ClipArea {
    RenderTaskCommonData common_data;
    float device_pixel_scale;
    vec2 screen_origin;
};

ClipArea fetch_clip_area(int index) {
    ClipArea area;

    if (index >= CLIP_TASK_EMPTY) {
        RectWithSize rect = RectWithSize(vec2(0.0), vec2(0.0));

        area.common_data = RenderTaskCommonData(rect, 0.0);
        area.device_pixel_scale = 0.0;
        area.screen_origin = vec2(0.0);
    } else {
        RenderTaskData task_data = fetch_render_task_data(index);

        area.common_data = task_data.common_data;
        area.device_pixel_scale = task_data.user_data.x;
        area.screen_origin = task_data.user_data.yz;
    }

    return area;
}

#endif //WR_VERTEX_SHADER
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

uniform HIGHP_SAMPLER_FLOAT sampler2D sGpuCache;

#define VECS_PER_IMAGE_RESOURCE     2

// TODO(gw): This is here temporarily while we have
//           both GPU store and cache. When the GPU
//           store code is removed, we can change the
//           PrimitiveInstance instance structure to
//           use 2x unsigned shorts as vertex attributes
//           instead of an int, and encode the UV directly
//           in the vertices.
ivec2 get_gpu_cache_uv(HIGHP_FS_ADDRESS int address) {
    return ivec2(uint(address) % WR_MAX_VERTEX_TEXTURE_WIDTH,
                 uint(address) / WR_MAX_VERTEX_TEXTURE_WIDTH);
}

vec4[2] fetch_from_gpu_cache_2_direct(ivec2 address) {
    return vec4[2](
        TEXEL_FETCH(sGpuCache, address, 0, ivec2(0, 0)),
        TEXEL_FETCH(sGpuCache, address, 0, ivec2(1, 0))
    );
}

vec4[2] fetch_from_gpu_cache_2(HIGHP_FS_ADDRESS int address) {
    ivec2 uv = get_gpu_cache_uv(address);
    return vec4[2](
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(0, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(1, 0))
    );
}

vec4 fetch_from_gpu_cache_1_direct(ivec2 address) {
    return texelFetch(sGpuCache, address, 0);
}

vec4 fetch_from_gpu_cache_1(HIGHP_FS_ADDRESS int address) {
    ivec2 uv = get_gpu_cache_uv(address);
    return texelFetch(sGpuCache, uv, 0);
}

#ifdef WR_VERTEX_SHADER

vec4[8] fetch_from_gpu_cache_8(int address) {
    ivec2 uv = get_gpu_cache_uv(address);
    return vec4[8](
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(0, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(1, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(2, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(3, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(4, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(5, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(6, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(7, 0))
    );
}

vec4[3] fetch_from_gpu_cache_3(int address) {
    ivec2 uv = get_gpu_cache_uv(address);
    return vec4[3](
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(0, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(1, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(2, 0))
    );
}

vec4[3] fetch_from_gpu_cache_3_direct(ivec2 address) {
    return vec4[3](
        TEXEL_FETCH(sGpuCache, address, 0, ivec2(0, 0)),
        TEXEL_FETCH(sGpuCache, address, 0, ivec2(1, 0)),
        TEXEL_FETCH(sGpuCache, address, 0, ivec2(2, 0))
    );
}

vec4[4] fetch_from_gpu_cache_4_direct(ivec2 address) {
    return vec4[4](
        TEXEL_FETCH(sGpuCache, address, 0, ivec2(0, 0)),
        TEXEL_FETCH(sGpuCache, address, 0, ivec2(1, 0)),
        TEXEL_FETCH(sGpuCache, address, 0, ivec2(2, 0)),
        TEXEL_FETCH(sGpuCache, address, 0, ivec2(3, 0))
    );
}

vec4[4] fetch_from_gpu_cache_4(int address) {
    ivec2 uv = get_gpu_cache_uv(address);
    return vec4[4](
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(0, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(1, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(2, 0)),
        TEXEL_FETCH(sGpuCache, uv, 0, ivec2(3, 0))
    );
}

//TODO: image resource is too specific for this module

struct ImageResource {
    RectWithEndpoint uv_rect;
    float layer;
    vec3 user_data;
};

ImageResource fetch_image_resource(int address) {
    //Note: number of blocks has to match `renderer::BLOCKS_PER_UV_RECT`
    vec4 data[2] = fetch_from_gpu_cache_2(address);
    RectWithEndpoint uv_rect = RectWithEndpoint(data[0].xy, data[0].zw);
    return ImageResource(uv_rect, data[1].x, data[1].yzw);
}

ImageResource fetch_image_resource_direct(ivec2 address) {
    vec4 data[2] = fetch_from_gpu_cache_2_direct(address);
    RectWithEndpoint uv_rect = RectWithEndpoint(data[0].xy, data[0].zw);
    return ImageResource(uv_rect, data[1].x, data[1].yzw);
}

// Fetch optional extra data for a texture cache resource. This can contain
// a polygon defining a UV rect within the texture cache resource.
// Note: the polygon coordinates are in homogeneous space.
struct ImageResourceExtra {
    vec4 st_tl;
    vec4 st_tr;
    vec4 st_bl;
    vec4 st_br;
};

ImageResourceExtra fetch_image_resource_extra(int address) {
    vec4 data[4] = fetch_from_gpu_cache_4(address + VECS_PER_IMAGE_RESOURCE);
    return ImageResourceExtra(
        data[0],
        data[1],
        data[2],
        data[3]
    );
}

#endif //WR_VERTEX_SHADER
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

flat varying vec4 vTransformBounds;

#ifdef WR_VERTEX_SHADER

#define VECS_PER_TRANSFORM   8U
uniform HIGHP_SAMPLER_FLOAT sampler2D sTransformPalette;

void init_transform_vs(vec4 local_bounds) {
    vTransformBounds = local_bounds;
}

struct Transform {
    mat4 m;
    mat4 inv_m;
    bool is_axis_aligned;
};

Transform fetch_transform(int id) {
    Transform transform;

    transform.is_axis_aligned = (id >> 24) == 0;
    int index = id & 0x00ffffff;

    // Create a UV base coord for each 8 texels.
    // This is required because trying to use an offset
    // of more than 8 texels doesn't work on some versions
    // of macOS.
    ivec2 uv = get_fetch_uv(index, VECS_PER_TRANSFORM);
    ivec2 uv0 = ivec2(uv.x + 0, uv.y);

    transform.m[0] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(0, 0));
    transform.m[1] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(1, 0));
    transform.m[2] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(2, 0));
    transform.m[3] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(3, 0));

    transform.inv_m[0] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(4, 0));
    transform.inv_m[1] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(5, 0));
    transform.inv_m[2] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(6, 0));
    transform.inv_m[3] = TEXEL_FETCH(sTransformPalette, uv0, 0, ivec2(7, 0));

    return transform;
}

// Return the intersection of the plane (set up by "normal" and "point")
// with the ray (set up by "ray_origin" and "ray_dir"),
// writing the resulting scaler into "t".
bool ray_plane(vec3 normal, vec3 pt, vec3 ray_origin, vec3 ray_dir, out float t)
{
    float denom = dot(normal, ray_dir);
    if (abs(denom) > 1e-6) {
        vec3 d = pt - ray_origin;
        t = dot(d, normal) / denom;
        return t >= 0.0;
    }

    return false;
}

// Apply the inverse transform "inv_transform"
// to the reference point "ref" in CSS space,
// producing a local point on a Transform plane,
// set by a base point "a" and a normal "n".
vec4 untransform(vec2 ref, vec3 n, vec3 a, mat4 inv_transform) {
    vec3 p = vec3(ref, -10000.0);
    vec3 d = vec3(0, 0, 1.0);

    float t = 0.0;
    // get an intersection of the Transform plane with Z axis vector,
    // originated from the "ref" point
    ray_plane(n, a, p, d, t);
    float z = p.z + d.z * t; // Z of the visible point on the Transform

    vec4 r = inv_transform * vec4(ref, z, 1.0);
    return r;
}

// Given a CSS space position, transform it back into the Transform space.
vec4 get_node_pos(vec2 pos, Transform transform) {
    // get a point on the scroll node plane
    vec4 ah = transform.m * vec4(0.0, 0.0, 0.0, 1.0);
    vec3 a = ah.xyz / ah.w;

    // get the normal to the scroll node plane
    vec3 n = transpose(mat3(transform.inv_m)) * vec3(0.0, 0.0, 1.0);
    return untransform(pos, n, a, transform.inv_m);
}

#endif //WR_VERTEX_SHADER

#ifdef WR_FRAGMENT_SHADER

float signed_distance_rect(vec2 pos, vec2 p0, vec2 p1) {
    vec2 d = max(p0 - pos, pos - p1);
    return length(max(vec2(0.0), d)) + min(0.0, max(d.x, d.y));
}

float init_transform_fs(vec2 local_pos) {
    // Get signed distance from local rect bounds.
    float d = signed_distance_rect(
        local_pos,
        vTransformBounds.xy,
        vTransformBounds.zw
    );

    // Find the appropriate distance to apply the AA smoothstep over.
    float aa_range = compute_aa_range(local_pos);

    // Only apply AA to fragments outside the signed distance field.
    return distance_aa(aa_range, d);
}

float init_transform_rough_fs(vec2 local_pos) {
    return point_inside_rect(
        local_pos,
        vTransformBounds.xy,
        vTransformBounds.zw
    );
}

#endif //WR_FRAGMENT_SHADER

#define EXTEND_MODE_CLAMP  0
#define EXTEND_MODE_REPEAT 1

#define SUBPX_DIR_NONE        0
#define SUBPX_DIR_HORIZONTAL  1
#define SUBPX_DIR_VERTICAL    2
#define SUBPX_DIR_MIXED       3

#define RASTER_LOCAL            0
#define RASTER_SCREEN           1

uniform sampler2DArray sPrevPassAlpha;
uniform sampler2DArray sPrevPassColor;

vec2 clamp_rect(vec2 pt, RectWithSize rect) {
    return clamp(pt, rect.p0, rect.p0 + rect.size);
}

// TODO: convert back to RectWithEndPoint if driver issues are resolved, if ever.
flat varying vec4 vClipMaskUvBounds;
// XY and W are homogeneous coordinates, Z is the layer index
varying vec4 vClipMaskUv;


#ifdef WR_VERTEX_SHADER

#define COLOR_MODE_FROM_PASS          0
#define COLOR_MODE_ALPHA              1
#define COLOR_MODE_SUBPX_CONST_COLOR  2
#define COLOR_MODE_SUBPX_BG_PASS0     3
#define COLOR_MODE_SUBPX_BG_PASS1     4
#define COLOR_MODE_SUBPX_BG_PASS2     5
#define COLOR_MODE_SUBPX_DUAL_SOURCE  6
#define COLOR_MODE_BITMAP             7
#define COLOR_MODE_COLOR_BITMAP       8
#define COLOR_MODE_IMAGE              9

uniform HIGHP_SAMPLER_FLOAT sampler2D sPrimitiveHeadersF;
uniform HIGHP_SAMPLER_FLOAT isampler2D sPrimitiveHeadersI;

// Instanced attributes
in ivec4 aData;

#define VECS_PER_PRIM_HEADER_F 2U
#define VECS_PER_PRIM_HEADER_I 2U

struct PrimitiveHeader {
    RectWithSize local_rect;
    RectWithSize local_clip_rect;
    float z;
    int specific_prim_address;
    int transform_id;
    ivec4 user_data;
};

PrimitiveHeader fetch_prim_header(int index) {
    PrimitiveHeader ph;

    ivec2 uv_f = get_fetch_uv(index, VECS_PER_PRIM_HEADER_F);
    vec4 local_rect = TEXEL_FETCH(sPrimitiveHeadersF, uv_f, 0, ivec2(0, 0));
    vec4 local_clip_rect = TEXEL_FETCH(sPrimitiveHeadersF, uv_f, 0, ivec2(1, 0));
    ph.local_rect = RectWithSize(local_rect.xy, local_rect.zw);
    ph.local_clip_rect = RectWithSize(local_clip_rect.xy, local_clip_rect.zw);

    ivec2 uv_i = get_fetch_uv(index, VECS_PER_PRIM_HEADER_I);
    ivec4 data0 = TEXEL_FETCH(sPrimitiveHeadersI, uv_i, 0, ivec2(0, 0));
    ivec4 data1 = TEXEL_FETCH(sPrimitiveHeadersI, uv_i, 0, ivec2(1, 0));
    ph.z = float(data0.x);
    ph.specific_prim_address = data0.y;
    ph.transform_id = data0.z;
    ph.user_data = data1;

    return ph;
}

struct VertexInfo {
    vec2 local_pos;
    vec2 snap_offset;
    vec4 world_pos;
};

VertexInfo write_vertex(RectWithSize instance_rect,
                        RectWithSize local_clip_rect,
                        float z,
                        Transform transform,
                        PictureTask task) {

    // Select the corner of the local rect that we are processing.
    vec2 local_pos = instance_rect.p0 + instance_rect.size * aPosition.xy;

    // Clamp to the two local clip rects.
    vec2 clamped_local_pos = clamp_rect(local_pos, local_clip_rect);

    // Transform the current vertex to world space.
    vec4 world_pos = transform.m * vec4(clamped_local_pos, 0.0, 1.0);

    // Convert the world positions to device pixel space.
    vec2 device_pos = world_pos.xy * task.device_pixel_scale;

    // Apply offsets for the render task to get correct screen location.
    vec2 final_offset = -task.content_origin + task.common_data.task_rect.p0;

    gl_Position = uTransform * vec4(device_pos + final_offset * world_pos.w, z * world_pos.w, world_pos.w);

    VertexInfo vi = VertexInfo(
        clamped_local_pos,
        vec2(0.0, 0.0),
        world_pos
    );

    return vi;
}

float cross2(vec2 v0, vec2 v1) {
    return v0.x * v1.y - v0.y * v1.x;
}

// Return intersection of line (p0,p1) and line (p2,p3)
vec2 intersect_lines(vec2 p0, vec2 p1, vec2 p2, vec2 p3) {
    vec2 d0 = p0 - p1;
    vec2 d1 = p2 - p3;

    float s0 = cross2(p0, p1);
    float s1 = cross2(p2, p3);

    float d = cross2(d0, d1);
    float nx = s0 * d1.x - d0.x * s1;
    float ny = s0 * d1.y - d0.y * s1;

    return vec2(nx / d, ny / d);
}

VertexInfo write_transform_vertex(RectWithSize local_segment_rect,
                                  RectWithSize local_prim_rect,
                                  RectWithSize local_clip_rect,
                                  vec4 clip_edge_mask,
                                  float z,
                                  Transform transform,
                                  PictureTask task) {
    // Calculate a clip rect from local_rect + local clip
    RectWithEndpoint clip_rect = to_rect_with_endpoint(local_clip_rect);
    RectWithEndpoint segment_rect = to_rect_with_endpoint(local_segment_rect);
    segment_rect.p0 = clamp(segment_rect.p0, clip_rect.p0, clip_rect.p1);
    segment_rect.p1 = clamp(segment_rect.p1, clip_rect.p0, clip_rect.p1);

    // Calculate a clip rect from local_rect + local clip
    RectWithEndpoint prim_rect = to_rect_with_endpoint(local_prim_rect);
    prim_rect.p0 = clamp(prim_rect.p0, clip_rect.p0, clip_rect.p1);
    prim_rect.p1 = clamp(prim_rect.p1, clip_rect.p0, clip_rect.p1);

    // As this is a transform shader, extrude by 2 (local space) pixels
    // in each direction. This gives enough space around the edge to
    // apply distance anti-aliasing. Technically, it:
    // (a) slightly over-estimates the number of required pixels in the simple case.
    // (b) might not provide enough edge in edge case perspective projections.
    // However, it's fast and simple. If / when we ever run into issues, we
    // can do some math on the projection matrix to work out a variable
    // amount to extrude.

    // Only extrude along edges where we are going to apply AA.
    float extrude_amount = 2.0;
    vec4 extrude_distance = vec4(extrude_amount) * clip_edge_mask;
    local_segment_rect.p0 -= extrude_distance.xy;
    local_segment_rect.size += extrude_distance.xy + extrude_distance.zw;

    // Select the corner of the local rect that we are processing.
    vec2 local_pos = local_segment_rect.p0 + local_segment_rect.size * aPosition.xy;

    // Convert the world positions to device pixel space.
    vec2 task_offset = task.common_data.task_rect.p0 - task.content_origin;

    // Transform the current vertex to the world cpace.
    vec4 world_pos = transform.m * vec4(local_pos, 0.0, 1.0);
    vec4 final_pos = vec4(
        world_pos.xy * task.device_pixel_scale + task_offset * world_pos.w,
        z * world_pos.w,
        world_pos.w
    );

    gl_Position = uTransform * final_pos;

    init_transform_vs(mix(
        vec4(prim_rect.p0, prim_rect.p1),
        vec4(segment_rect.p0, segment_rect.p1),
        clip_edge_mask
    ));

    VertexInfo vi = VertexInfo(
        local_pos,
        vec2(0.0),
        world_pos
    );

    return vi;
}

void write_clip(vec4 world_pos, vec2 snap_offset, ClipArea area) {
    vec2 uv = world_pos.xy * area.device_pixel_scale +
        world_pos.w * (snap_offset + area.common_data.task_rect.p0 - area.screen_origin);
    vClipMaskUvBounds = vec4(
        area.common_data.task_rect.p0,
        area.common_data.task_rect.p0 + area.common_data.task_rect.size
    );
    vClipMaskUv = vec4(uv, area.common_data.texture_layer_index, world_pos.w);
}

// Read the exta image data containing the homogeneous screen space coordinates
// of the corners, interpolate between them, and return real screen space UV.
vec2 get_image_quad_uv(int address, vec2 f) {
    ImageResourceExtra extra_data = fetch_image_resource_extra(address);
    vec4 x = mix(extra_data.st_tl, extra_data.st_tr, f.x);
    vec4 y = mix(extra_data.st_bl, extra_data.st_br, f.x);
    vec4 z = mix(x, y, f.y);
    return z.xy / z.w;
}
#endif //WR_VERTEX_SHADER

#ifdef WR_FRAGMENT_SHADER

float do_clip() {
    // check for the dummy bounds, which are given to the opaque objects
    if (vClipMaskUvBounds.xy == vClipMaskUvBounds.zw) {
        return 1.0;
    }
    // anything outside of the mask is considered transparent
    //Note: we assume gl_FragCoord.w == interpolated(1 / vClipMaskUv.w)
    vec2 mask_uv = vClipMaskUv.xy * gl_FragCoord.w;
    bvec2 left = lessThanEqual(vClipMaskUvBounds.xy, mask_uv); // inclusive
    bvec2 right = greaterThan(vClipMaskUvBounds.zw, mask_uv); // non-inclusive
    // bail out if the pixel is outside the valid bounds
    if (!all(bvec4(left, right))) {
        return 0.0;
    }
    // finally, the slow path - fetch the mask value from an image
    // Note the Z getting rounded to the nearest integer because the variable
    // is still interpolated and becomes a subject of precision-caused
    // fluctuations, see https://bugzilla.mozilla.org/show_bug.cgi?id=1491911
    ivec3 tc = ivec3(mask_uv, vClipMaskUv.z + 0.5);
    return texelFetch(sPrevPassAlpha, tc, 0).r;
}

#ifdef WR_FEATURE_DITHERING
vec4 dither(vec4 color) {
    const int matrix_mask = 7;

    ivec2 pos = ivec2(gl_FragCoord.xy) & ivec2(matrix_mask);
    float noise_normalized = (texelFetch(sDither, pos, 0).r * 255.0 + 0.5) / 64.0;
    float noise = (noise_normalized - 0.5) / 256.0; // scale down to the unit length

    return color + vec4(noise, noise, noise, 0);
}
#else
vec4 dither(vec4 color) {
    return color;
}
#endif //WR_FEATURE_DITHERING

vec4 sample_gradient(HIGHP_FS_ADDRESS int address, float offset, float gradient_repeat) {
    // Modulo the offset if the gradient repeats.
    float x = mix(offset, fract(offset), gradient_repeat);

    // Calculate the color entry index to use for this offset:
    //     offsets < 0 use the first color entry, 0
    //     offsets from [0, 1) use the color entries in the range of [1, N-1)
    //     offsets >= 1 use the last color entry, N-1
    //     so transform the range [0, 1) -> [1, N-1)

    // TODO(gw): In the future we might consider making the size of the
    // LUT vary based on number / distribution of stops in the gradient.
    const int GRADIENT_ENTRIES = 128;
    x = 1.0 + x * float(GRADIENT_ENTRIES);

    // Calculate the texel to index into the gradient color entries:
    //     floor(x) is the gradient color entry index
    //     fract(x) is the linear filtering factor between start and end
    int lut_offset = 2 * int(floor(x));     // There is a [start, end] color per entry.

    // Ensure we don't fetch outside the valid range of the LUT.
    lut_offset = clamp(lut_offset, 0, 2 * (GRADIENT_ENTRIES + 1));

    // Fetch the start and end color.
    vec4 texels[2] = fetch_from_gpu_cache_2(address + lut_offset);

    // Finally interpolate and apply dithering
    return dither(mix(texels[0], texels[1], fract(x)));
}

#endif //WR_FRAGMENT_SHADER

flat varying vec4 vColor;
varying vec3 vUv;
flat varying vec4 vUvBorder;
flat varying vec2 vMaskSwizzle;

#ifdef WR_FEATURE_GLYPH_TRANSFORM
varying vec4 vUvClip;
#endif

#ifdef WR_VERTEX_SHADER

#define VECS_PER_TEXT_RUN           2
#define GLYPHS_PER_GPU_BLOCK        2U

#ifdef WR_FEATURE_GLYPH_TRANSFORM
RectWithSize transform_rect(RectWithSize rect, mat2 transform) {
    vec2 center = transform * (rect.p0 + rect.size * 0.5);
    vec2 radius = mat2(abs(transform[0]), abs(transform[1])) * (rect.size * 0.5);
    return RectWithSize(center - radius, radius * 2.0);
}

bool rect_inside_rect(RectWithSize little, RectWithSize big) {
    return all(lessThanEqual(vec4(big.p0, little.p0 + little.size),
                             vec4(little.p0, big.p0 + big.size)));
}
#endif //WR_FEATURE_GLYPH_TRANSFORM

struct Glyph {
    vec2 offset;
};

Glyph fetch_glyph(int specific_prim_address,
                  int glyph_index) {
    // Two glyphs are packed in each texel in the GPU cache.
    int glyph_address = specific_prim_address +
                        VECS_PER_TEXT_RUN +
                        int(uint(glyph_index) / GLYPHS_PER_GPU_BLOCK);
    vec4 data = fetch_from_gpu_cache_1(glyph_address);
    // Select XY or ZW based on glyph index.
    // We use "!= 0" instead of "== 1" here in order to work around a driver
    // bug with equality comparisons on integers.
    vec2 glyph = mix(data.xy, data.zw,
                     bvec2(uint(glyph_index) % GLYPHS_PER_GPU_BLOCK != 0U));

    return Glyph(glyph);
}

struct GlyphResource {
    vec4 uv_rect;
    float layer;
    vec2 offset;
    float scale;
};

GlyphResource fetch_glyph_resource(int address) {
    vec4 data[2] = fetch_from_gpu_cache_2(address);
    return GlyphResource(data[0], data[1].x, data[1].yz, data[1].w);
}

struct TextRun {
    vec4 color;
    vec4 bg_color;
};

TextRun fetch_text_run(int address) {
    vec4 data[2] = fetch_from_gpu_cache_2(address);
    return TextRun(data[0], data[1]);
}

VertexInfo write_text_vertex(RectWithSize local_clip_rect,
                             float z,
                             int raster_space,
                             Transform transform,
                             PictureTask task,
                             vec2 text_offset,
                             vec2 glyph_offset,
                             RectWithSize glyph_rect,
                             vec2 snap_bias) {
    // The offset to snap the glyph rect to a device pixel
    vec2 snap_offset = vec2(0.0);
    // Transform from glyph space to local space
    mat2 glyph_transform_inv = mat2(1.0);

#ifdef WR_FEATURE_GLYPH_TRANSFORM
    bool remove_subpx_offset = true;
#else
    bool remove_subpx_offset = transform.is_axis_aligned;
#endif
    // Compute the snapping offset only if the scroll node transform is axis-aligned.
    if (remove_subpx_offset) {
        // Be careful to only snap with the transform when in screen raster space.
        switch (raster_space) {
            case RASTER_SCREEN: {
                // Transform from local space to glyph space.
                float device_scale = task.device_pixel_scale / transform.m[3].w;
                mat2 glyph_transform = mat2(transform.m) * device_scale;

                // Ensure the transformed text offset does not contain a subpixel translation
                // such that glyph snapping is stable for equivalent glyph subpixel positions.
                vec2 device_text_pos = glyph_transform * text_offset + transform.m[3].xy * device_scale;
                snap_offset = floor(device_text_pos + 0.5) - device_text_pos;

                // Snap the glyph offset to a device pixel, using an appropriate bias depending
                // on whether subpixel positioning is required.
                vec2 device_glyph_offset = glyph_transform * glyph_offset;
                snap_offset += floor(device_glyph_offset + snap_bias) - device_glyph_offset;

                // Transform from glyph space back to local space.
                glyph_transform_inv = inverse(glyph_transform);

#ifndef WR_FEATURE_GLYPH_TRANSFORM
                // If not using transformed subpixels, the glyph rect is actually in local space.
                // So convert the snap offset back to local space.
                snap_offset = glyph_transform_inv * snap_offset;
#endif
                break;
            }
            default: {
                // Otherwise, when in local raster space, the transform may be animated, so avoid
                // snapping with the transform to avoid oscillation.
                snap_offset = floor(text_offset + 0.5) - text_offset;
                snap_offset += floor(glyph_offset + snap_bias) - glyph_offset;
                break;
            }
        }
    }

    // Actually translate the glyph rect to a device pixel using the snap offset.
    glyph_rect.p0 += snap_offset;

#ifdef WR_FEATURE_GLYPH_TRANSFORM
    // The glyph rect is in device space, so transform it back to local space.
    RectWithSize local_rect = transform_rect(glyph_rect, glyph_transform_inv);

    // Select the corner of the glyph's local space rect that we are processing.
    vec2 local_pos = local_rect.p0 + local_rect.size * aPosition.xy;

    // If the glyph's local rect would fit inside the local clip rect, then select a corner from
    // the device space glyph rect to reduce overdraw of clipped pixels in the fragment shader.
    // Otherwise, fall back to clamping the glyph's local rect to the local clip rect.
    if (rect_inside_rect(local_rect, local_clip_rect)) {
        local_pos = glyph_transform_inv * (glyph_rect.p0 + glyph_rect.size * aPosition.xy);
    }
#else
    // Select the corner of the glyph rect that we are processing.
    vec2 local_pos = glyph_rect.p0 + glyph_rect.size * aPosition.xy;
#endif

    // Clamp to the local clip rect.
    local_pos = clamp_rect(local_pos, local_clip_rect);

    // Map the clamped local space corner into device space.
    vec4 world_pos = transform.m * vec4(local_pos, 0.0, 1.0);
    vec2 device_pos = world_pos.xy * task.device_pixel_scale;

    // Apply offsets for the render task to get correct screen location.
    vec2 final_offset = -task.content_origin + task.common_data.task_rect.p0;

    gl_Position = uTransform * vec4(device_pos + final_offset * world_pos.w, z * world_pos.w, world_pos.w);

    VertexInfo vi = VertexInfo(
        local_pos,
        snap_offset,
        world_pos
    );

    return vi;
}

void main(void) {
    int prim_header_address = aData.x;
    int glyph_index = aData.y & 0xffff;
    int render_task_index = aData.y >> 16;
    int resource_address = aData.z;
    int raster_space = aData.w >> 16;
    int subpx_dir = (aData.w >> 8) & 0xff;
    int color_mode = aData.w & 0xff;

    PrimitiveHeader ph = fetch_prim_header(prim_header_address);
    Transform transform = fetch_transform(ph.transform_id);
    ClipArea clip_area = fetch_clip_area(ph.user_data.w);
    PictureTask task = fetch_picture_task(render_task_index);

    TextRun text = fetch_text_run(ph.specific_prim_address);
    vec2 text_offset = vec2(ph.user_data.xy) / 256.0;

    if (color_mode == COLOR_MODE_FROM_PASS) {
        color_mode = uMode;
    }

    Glyph glyph = fetch_glyph(ph.specific_prim_address, glyph_index);
    glyph.offset += ph.local_rect.p0 - text_offset;

    GlyphResource res = fetch_glyph_resource(resource_address);

#ifdef WR_FEATURE_GLYPH_TRANSFORM
    // Transform from local space to glyph space.
    mat2 glyph_transform = mat2(transform.m) * task.device_pixel_scale;

    // Compute the glyph rect in glyph space.
    RectWithSize glyph_rect = RectWithSize(res.offset + glyph_transform * (text_offset + glyph.offset),
                                           res.uv_rect.zw - res.uv_rect.xy);
#else
    float raster_scale = float(ph.user_data.z) / 65535.0;

    // Scale from glyph space to local space.
    float scale = res.scale / (raster_scale * task.device_pixel_scale);

    // Compute the glyph rect in local space.
    RectWithSize glyph_rect = RectWithSize(scale * res.offset + text_offset + glyph.offset,
                                           scale * (res.uv_rect.zw - res.uv_rect.xy));
#endif

    vec2 snap_bias;
    // In subpixel mode, the subpixel offset has already been
    // accounted for while rasterizing the glyph. However, we
    // must still round with a subpixel bias rather than rounding
    // to the nearest whole pixel, depending on subpixel direciton.
    switch (subpx_dir) {
        case SUBPX_DIR_NONE:
        default:
            snap_bias = vec2(0.5);
            break;
        case SUBPX_DIR_HORIZONTAL:
            // Glyphs positioned [-0.125, 0.125] get a
            // subpx position of zero. So include that
            // offset in the glyph position to ensure
            // we round to the correct whole position.
            snap_bias = vec2(0.125, 0.5);
            break;
        case SUBPX_DIR_VERTICAL:
            snap_bias = vec2(0.5, 0.125);
            break;
        case SUBPX_DIR_MIXED:
            snap_bias = vec2(0.125);
            break;
    }

    VertexInfo vi = write_text_vertex(ph.local_clip_rect,
                                      ph.z,
                                      raster_space,
                                      transform,
                                      task,
                                      text_offset,
                                      glyph.offset,
                                      glyph_rect,
                                      snap_bias);
    glyph_rect.p0 += vi.snap_offset;

#ifdef WR_FEATURE_GLYPH_TRANSFORM
    vec2 f = (glyph_transform * vi.local_pos - glyph_rect.p0) / glyph_rect.size;
    vUvClip = vec4(f, 1.0 - f);
#else
    vec2 f = (vi.local_pos - glyph_rect.p0) / glyph_rect.size;
#endif

    write_clip(vi.world_pos, vi.snap_offset, clip_area);

    switch (color_mode) {
        case COLOR_MODE_ALPHA:
        case COLOR_MODE_BITMAP:
            vMaskSwizzle = vec2(0.0, 1.0);
            vColor = text.color;
            break;
        case COLOR_MODE_SUBPX_BG_PASS2:
        case COLOR_MODE_SUBPX_DUAL_SOURCE:
            vMaskSwizzle = vec2(1.0, 0.0);
            vColor = text.color;
            break;
        case COLOR_MODE_SUBPX_CONST_COLOR:
        case COLOR_MODE_SUBPX_BG_PASS0:
        case COLOR_MODE_COLOR_BITMAP:
            vMaskSwizzle = vec2(1.0, 0.0);
            vColor = vec4(text.color.a);
            break;
        case COLOR_MODE_SUBPX_BG_PASS1:
            vMaskSwizzle = vec2(-1.0, 1.0);
            vColor = vec4(text.color.a) * text.bg_color;
            break;
        default:
            vMaskSwizzle = vec2(0.0);
            vColor = vec4(1.0);
    }

    vec2 texture_size = vec2(textureSize(sColor0, 0));
    vec2 st0 = res.uv_rect.xy / texture_size;
    vec2 st1 = res.uv_rect.zw / texture_size;

    vUv = vec3(mix(st0, st1, f), res.layer);
    vUvBorder = (res.uv_rect + vec4(0.5, 0.5, -0.5, -0.5)) / texture_size.xyxy;
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
    vec3 tc = vec3(clamp(vUv.xy, vUvBorder.xy, vUvBorder.zw), vUv.z);
    vec4 mask = texture(sColor0, tc);
    mask.rgb = mask.rgb * vMaskSwizzle.x + mask.aaa * vMaskSwizzle.y;

    float alpha = do_clip();
#ifdef WR_FEATURE_GLYPH_TRANSFORM
    alpha *= float(all(greaterThanEqual(vUvClip, vec4(0.0))));
#endif

#if defined(WR_FEATURE_DEBUG_OVERDRAW)
    oFragColor = WR_DEBUG_OVERDRAW_COLOR;
#elif defined(WR_FEATURE_DUAL_SOURCE_BLENDING)
    vec4 alpha_mask = mask * alpha;
    oFragColor = vColor * alpha_mask;
    oFragBlend = alpha_mask * vColor.a;
#else
    write_output(vColor * mask * alpha);
#endif
}
#endif

    "#;


    let mut glc = unsafe { gl::GlFns::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _) };
    let gl = ErrorCheckingGl::wrap(glc);// Rc::get_mut(&mut glc).unwrap();

    let shader_program = init_shader_program(&gl, vs_source, fs_source);

    let vs_source = include_bytes!("ps_text_run.vert.pp");
    let fs_source = include_bytes!("ps_text_run.frag.pp");

    let shader_program = init_shader_program(&gl, vs_source, fs_source);


    let vs_source = include_bytes!("ps_text_run.vert.opt");
    let fs_source = include_bytes!("ps_text_run.frag.opt");

    let shader_program = init_shader_program(&gl, vs_source, fs_source);
    panic!();

    let vertex_position = gl.get_attrib_location(shader_program, "a_vertex_position");
    let texture_coord = gl.get_attrib_location(shader_program, "a_texture_coord");

    let projection_matrix_loc = gl.get_uniform_location(shader_program, "u_projection_matrix");
    let model_view_matrix_loc = gl.get_uniform_location(shader_program, "u_model_view_matrix");
    let u_sampler = gl.get_uniform_location(shader_program, "u_sampler");


    let mut image = load_image(texture_src_type);
    let buffers = init_buffers(&gl, texture_rectangle, image.width, image.height);


    let texture = load_texture(&gl, &image,
                               texture_target,
                               texture_internal_format,
                               texture_src_format,
                               texture_src_type,
                               &options);

    let vao = gl.gen_vertex_arrays(1)[0];
    gl.bind_vertex_array(vao);


    let mut running = true;
    let mut cube_rotation: f32 = 0.;
    while running {
        events_loop.poll_events(|event| {
            match event {
                glutin::Event::WindowEvent{ event, .. } => match event {
                    glutin::WindowEvent::Closed => running = false,
                    glutin::WindowEvent::Resized(w, h) => gl_window.resize(w, h),
                    _ => ()
                },
                _ => ()
            }
        });


        // Bind the texture to texture unit 0
        gl.bind_texture(texture_target, texture.id);

        {
            let level = 0;
            if options.pbo {
                let id = texture.pbo.unwrap();
                gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, id);
                if true {
                    //gl.buffer_sub_data_untyped(gl::PIXEL_UNPACK_BUFFER, 0, (image.width * image.height * 4) as _, image.data[..].as_ptr() as *const libc::c_void);
                    gl.buffer_data_untyped(gl::PIXEL_UNPACK_BUFFER, (image.width * image.height * bpp(texture_src_type)) as _, image.data[..].as_ptr() as *const libc::c_void, gl::DYNAMIC_DRAW);

                } else {
                    //gl.map_buffer(gl::PIXEL_UNPACK_BUFFER, gl::WRITE_ONLY);
                }
                if options.texture_array {
                    gl.tex_sub_image_3d_pbo(
                        texture_target,
                        level,
                        0,
                        0,
                        0,
                        image.width,
                        image.height,
                        1,
                        texture_src_format,
                        texture_src_type,
                        0,
                    );
                } else {
                    gl.tex_sub_image_2d_pbo(
                        texture_target,
                        level,
                        0,
                        0,
                        image.width,
                        image.height,
                        texture_src_format,
                        texture_src_type,
                        0,
                    );
                }
                gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);
            } else {
                if options.client_storage {
                    //gl.tex_parameter_i(texture_target, gl::TEXTURE_STORAGE_HINT_APPLE, gl::STORAGE_CACHED_APPLE as gl::GLint);
                    //gl.pixel_store_i(gl::UNPACK_CLIENT_STORAGE_APPLE, true as gl::GLint);
                }
                gl.tex_sub_image_2d(texture_target, level, 0, 0, image.width, image.height, texture_src_format, texture_src_type, &image.data[..]);
                // sub image uploads are still fast as long as the memory is in the same place
                //gl.tex_sub_image_2d(texture_target, level, 0, 256, image.width, 4096, texture_src_format, texture_src_type, &image.data[image.width as usize *4*256..image.width as usize *4*4096]);

            }
        }


        gl.clear_color(1., 0., 0., 1.);
        gl.clear_depth(1.);
        gl.enable(gl::DEPTH_TEST);
        gl.depth_func(gl::LEQUAL);

        gl.clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

        // Create a perspective matrix, a special matrix that is
        // used to simulate the distortion of perspective in a camera.
        // Our field of view is 45 degrees, with a width/height
        // ratio that matches the display size of the canvas
        // and we only want to see objects between 0.1 units
        // and 100 units away from the camera.

        let field_of_view = 45. * std::f32::consts::PI / 180.;   // in radians
        let width = 1024.;
        let height = 768.;
        let aspect = width / height;
        let z_near = 0.1;
        let z_far = 100.0;

        let fovy = field_of_view;
        let near = z_near;
        let far = z_far;
        let f = 1. / (fovy / 2.).tan();
        let nf = 1. / (near - far);

        let projection_matrix = Transform3D::<f32>::row_major(
            f / aspect, 0., 0., 0.,
            0.,  f, 0., 0.,
            0., 0.,  (far + near) * nf, -1.,
            0., 0., 2. * far * near * nf, 0.
        );

        let mut model_view_matrix = Transform3D::<f32>::identity();
        model_view_matrix = model_view_matrix.post_translate(Vector3D::new(-0., 0., -6.0));
        model_view_matrix = model_view_matrix.pre_rotate(0., 0., 1., euclid::Angle::radians(cube_rotation));
        model_view_matrix = model_view_matrix.pre_rotate(0., 1., 0., euclid::Angle::radians(cube_rotation * 0.7));

        {
            let num_components = 3;
            let ty = gl::FLOAT;
            let normalize = false;
            let stride = 0;
            let offset = 0;
            gl.bind_buffer(gl::ARRAY_BUFFER, buffers.position);
            gl.vertex_attrib_pointer(vertex_position as u32, num_components, ty, normalize, stride, offset);
            gl.enable_vertex_attrib_array(vertex_position as u32);
        }

        {
            let num_components = 2;
            let ty = gl::FLOAT;
            let normalize = false;
            let stride = 0;
            let offset = 0;
            gl.bind_buffer(gl::ARRAY_BUFFER, buffers.texture_coord);
            gl.vertex_attrib_pointer(texture_coord as u32, num_components, ty, normalize, stride, offset);
            gl.enable_vertex_attrib_array(texture_coord as u32);
        }

        gl.bind_buffer(gl::ELEMENT_ARRAY_BUFFER, buffers.indices);

        gl.use_program(shader_program);

        gl.uniform_matrix_4fv(
            projection_matrix_loc,
            false,
            &projection_matrix.to_row_major_array());

        gl.uniform_matrix_4fv(
            model_view_matrix_loc,
            false,
            &model_view_matrix.to_row_major_array());

        // Specify the texture to map onto the faces.

        // Tell OpenGL we want to affect texture unit 0
        gl.active_texture(gl::TEXTURE0);


        gl.uniform_1i(u_sampler, 0);

        {
            let vertex_count = 36;
            let ty = gl::UNSIGNED_SHORT;
            let offset = 0;
            gl.draw_elements(gl::TRIANGLES, vertex_count, ty, offset);
        }
        //paint_square(&mut image);
        gl_window.swap_buffers().unwrap();


        cube_rotation += 0.1;
    }
}
