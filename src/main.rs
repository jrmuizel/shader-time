extern crate gleam;
extern crate glutin;
extern crate libc;
extern crate euclid;

use euclid::{Transform3D, Vector3D};

use gleam::gl;
use glutin::GlContext;

use std::rc::Rc;

use gleam::gl::GLuint;


fn init_shader_program(gl: &mut gl::Gl, vs_source: &[u8], fs_source: &[u8]) -> gl::GLuint {
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
    shader_program
}

struct Buffers {
    position: GLuint,
    texture_coord: GLuint,
    indices: GLuint
}

fn init_buffers(gl: &mut gl::Gl) -> Buffers {
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


    let texture_coordinates = [
        // Front
        0.0f32,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,
        // Back
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,
        // Top
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,
        // Bottom
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,
        // Right
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,
        // Left
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,
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

fn load_texture(gl: &mut gl::Gl) -> GLuint {
    let texture = gl.gen_textures(1)[0];

    gl.bind_texture(gl::TEXTURE_2D, texture);

    let decoder = png::Decoder::new(std::fs::File::open("cubetexture.png").unwrap());
    let (info, mut reader) = decoder.read_info().unwrap();
    // Allocate the output buffer.
    let mut buf = vec![0; info.buffer_size()];
    // Read the next frame. Currently this function should only called once.
    // The default options
    reader.next_frame(&mut buf).unwrap();
    let level = 0;
    let internal_format = gl::RGBA;
    let src_format = gl::RGBA;
    let src_type = gl::UNSIGNED_BYTE;
    let border = 0;
    gl.tex_image_2d(gl::TEXTURE_2D, level, internal_format as i32, info.width as i32, info.height as i32, border, src_format, src_type, Some(&buf[..]));

    gl.tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
    texture
}


fn load_shader(gl: &mut gl::Gl, shader_type: gl::GLenum, source: &[u8]) -> gl::GLuint {
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

    let vs_source = b"
    #version 140

    in vec4 a_vertex_position;
    in vec2 a_texture_coord;
    uniform mat4 u_model_view_matrix;
    uniform mat4 u_projection_matrix;
    out vec2 v_texture_coord;
    void main(void) {
        gl_Position = u_projection_matrix * u_model_view_matrix * a_vertex_position;
        v_texture_coord = a_texture_coord;
    }";

    let fs_source = b"
    #version 140

    in vec2 v_texture_coord;
    uniform sampler2D u_sampler;
    out vec4 fragment_color;
    void main(void) {
        fragment_color = texture(u_sampler, v_texture_coord);
    }
    ";

    let mut glc = unsafe { gl::GlFns::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _) };
    let gl = Rc::get_mut(&mut glc).unwrap();

    let shader_program = init_shader_program(gl, vs_source, fs_source);


    let vertex_position = gl.get_attrib_location(shader_program, "a_vertex_position");
    let texture_coord = gl.get_attrib_location(shader_program, "a_texture_coord");

    let projection_matrix_loc = gl.get_uniform_location(shader_program, "u_projection_matrix");
    let model_view_matrix_loc = gl.get_uniform_location(shader_program, "u_model_view_matrix");
    let u_sampler = gl.get_uniform_location(shader_program, "u_sampler");

    let buffers = init_buffers(gl);

    let texture = load_texture(gl);

    let vao = gl.gen_vertex_arrays(1)[0];
    gl.bind_vertex_array(vao);




    /*
    let texture_count = 14;

    let data = [0u8; 30000];

    let TEXTURE_WIDTH = 800;
    let TEXTURE_HEIGHT = 800;

    let i = 0;
    let k = gl.texture_range_apple(gl::TEXTURE_RECTANGLE_ARB, &data[..]);

    let tex_offset = 0;
    let target = gl::TEXTURE_2D;
    let textures = gl.gen_textures(texture_count);
    for tex in &textures {
        gl.bind_texture(target, *tex);
        gl.tex_parameter_i(target, gl::TEXTURE_STORAGE_HINT_APPLE , gl::STORAGE_CACHED_APPLE as gl::GLint);
        gl.pixel_store_i(gl::UNPACK_CLIENT_STORAGE_APPLE, true as gl::GLint);

        // Rectangle textures has its limitations compared to using POT textures, for example,
        // Rectangle textures can't use mipmap filtering
        gl.tex_parameter_i(target, gl::TEXTURE_MIN_FILTER, gl::NEAREST as gl::GLint);
        gl.tex_parameter_i(target, gl::TEXTURE_MAG_FILTER, gl::NEAREST as gl::GLint);

        // Rectangle textures can't use the GL_REPEAT warp mode
        gl.tex_parameter_i(target, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as gl::GLint);
        gl.tex_parameter_i(target, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::GLint);

        gl.pixel_store_i(gl::UNPACK_ROW_LENGTH, 0);

        gl.tex_image_2d(target, 0, gl::RGBA as gl::GLint, TEXTURE_WIDTH, TEXTURE_HEIGHT, 0,
                        gl::BGRA, gl::UNSIGNED_INT_8_8_8_8_REV,
                        Some(&data[(TEXTURE_WIDTH * TEXTURE_HEIGHT * 4 * (i + tex_offset)) as usize..]));



    }*/

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


        /*

        for tex in &textures {
      /*n tex_buffer(&self, target: GLenum, internal_format: GLenum, buffer: GLuint) {
      unsafe {
          self.ffi_gl_.TexBuffer(target, internal_format, buffer);
      }
  }*/
      gl.bind_texture(gl::TEXTURE_RECTANGLE_ARB, *tex);


      gl.tex_parameter_i(target, gl::TEXTURE_STORAGE_HINT_APPLE, gl::STORAGE_CACHED_APPLE as gl::GLint);
      gl.pixel_store_i(gl::UNPACK_CLIENT_STORAGE_APPLE, true as gl::GLint);
      gl.tex_sub_image_2d(target, 0, 0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT,
                          gl::BGRA, gl::UNSIGNED_INT_8_8_8_8_REV,
                          &data[(TEXTURE_WIDTH * TEXTURE_HEIGHT * 4 * (i + tex_offset)) as usize..]);
  }

  for tex in &textures {
      /*n tex_buffer(&self, target: GLenum, internal_format: GLenum, buffer: GLuint) {
      unsafe {
          self.ffi_gl_.TexBuffer(target, internal_format, buffer);
      }
  }*/
      /*
      gl.bind_texture(gl::TEXTURE_RECTANGLE_ARB, *tex);


      gl.tex_parameter_i(gl::TEXTURE_RECTANGLE_ARB, gl::TEXTURE_STORAGE_HINT_APPLE, gl::STORAGE_CACHED_APPLE as gl::GLint);
      gl.pixel_store_i(gl::UNPACK_CLIENT_STORAGE_APPLE, true as gl::GLint);
      gl.tex_sub_image_2d(gl::TEXTURE_RECTANGLE_ARB, 0, 0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT,
                          gl::BGRA, gl::UNSIGNED_INT_8_8_8_8_REV,
                          &data[(TEXTURE_WIDTH * TEXTURE_HEIGHT * 4 * (i + tex_offset)) as usize..]);*/
  }*/

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

        // Bind the texture to texture unit 0
        gl.bind_texture(gl::TEXTURE_2D, texture);

        gl.uniform_1i(u_sampler, 0);

        {
            let vertex_count = 36;
            let ty = gl::UNSIGNED_SHORT;
            let offset = 0;
            gl.draw_elements(gl::TRIANGLES, vertex_count, ty, offset);
        }

        gl_window.swap_buffers().unwrap();
        cube_rotation += 0.1;
    }
}
