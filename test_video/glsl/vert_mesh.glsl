#version 330 core
// Uniforms
// ------------------------------------
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
uniform   vec4 u_color;
uniform  float u_rotation_parts[27];
uniform   mat4 u_rotation_matrix;

// Attributes
// ------------------------------------
attribute vec4 a_position;
attribute vec3 a_normal;
attribute vec4 a_color;
attribute vec2 a_texcoord;
attribute float a_mtl;
attribute float a_group;

// Varying
// ------------------------------------
out vec4 v_color;
out vec3 FragPos;
out vec3 Normal;
varying vec2 v_texcoord;
flat out float v_mtl;
flat out float v_group;
void main()
{
    v_color = a_color;// * u_color;
    // gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    if (int(u_rotation_parts[int(a_group)])==0)
    {
        gl_Position = u_projection * u_view * u_model * a_position;
        FragPos = vec3(u_view * u_model * a_position);
        Normal = mat3(transpose(inverse(u_view * u_model))) * a_normal;
    }
    else
    {
        gl_Position = u_projection * u_view * u_model * u_rotation_matrix * a_position;
        FragPos = vec3(u_view * u_model * u_rotation_matrix * a_position);
        Normal = mat3(transpose(inverse(u_view * u_model * u_rotation_matrix))) * a_normal;
    }
    v_texcoord = a_texcoord;
    v_mtl = a_mtl;
    v_group = a_group;
}