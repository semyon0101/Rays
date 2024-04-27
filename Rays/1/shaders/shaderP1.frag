#version 450
//#extension GL_GOOGLE_include_directive : enable
#include "hg_sdf.glsl"

layout (set = 0, binding = 0) uniform ParameterUBO {
    int width;
    int height;
} ubo;

const float FOV = 1.0;
const int MAX_STEPS = 256;
const float MAX_DIST = 500;
const float EPSILON = 0.001;

vec2 map(vec3 p) {
    pMod3(p, vec3(5));


    float sphereDist = fSphere(p, 1.0);
    float sphereID = 1.0;
    vec2 sphere = vec2(sphereDist, sphereID);

    vec2 res = sphere;
    return res;
}

vec2 rayMarch(vec3 ro, vec3 rd) {
    vec2 hit = vec2(0);
    vec2 object = vec2(0);
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + object.x * rd;
        hit = map(p);
        object.x += hit.x;
        object.y = hit.y;
        if (abs(hit.x) < EPSILON || object.x > MAX_DIST) break;
    
    }
    return object;
}

vec3 render(vec2 uv) {
    vec3 col = vec3(0);

    vec3 ro = vec3(0, 0, -3);
    vec3 rd = normalize(vec3(uv,FOV));

    vec2 object = rayMarch(ro, rd);

    if (object.x < MAX_DIST) {
        col += 3.0 /object.x;
    }


    return col;
}

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = (2*gl_FragCoord.xy-vec2(ubo.width, ubo.height))/vec2(ubo.width, ubo.height)/2;
    float d = PI;
    vec3 col = render(uv);
    outColor = vec4(col, 1.0);
}