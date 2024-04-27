#version 450

layout (set = 0, binding = 0) uniform ParameterUBO {
    int width;
    int height;
} ubo;


vec2 positions[6] = vec2[](
    vec2(-1, 1),
    vec2(1, 1),
    vec2(1, -1),
    vec2(-1, -1),
    vec2(-1, 1),
    vec2(1, -1)
);


void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}