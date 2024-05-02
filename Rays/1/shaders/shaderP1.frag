#version 450
//#extension GL_GOOGLE_include_directive : enable
#include "hg_sdf.glsl"

layout (set = 0, binding = 0) uniform ParameterUBO {
    vec3 position;
    vec3 direction;
    float fov;
    int width;
    int height;
} ubo;

const int MAX_STEPS = 256;
const float MAX_DIST = 500;
const float EPSILON = 0.001;

vec2 fOpUnionID(vec2 res1, vec2 res2) {
    return (res1.x < res2.x) ? res1 : res2;
}

vec2 map(vec3 p) {
    float planeDist = fPlane(p, vec3(0, 1, 0), 1.0);
    float planeID = 2.0;
    vec2 plane = vec2(planeDist, planeID);

    

    float sphereDist = fSphere(p, 1);
    float sphereID = 1.0;
    vec2 sphere = vec2(sphereDist, sphereID);

    vec2 res = fOpUnionID(sphere, plane);
    return res;
}

vec3 getMaterial(vec3 p, float id) { 
    vec3 m;
    switch (int(id)) {
        case 1:
        m = vec3(0.9, 0.0, 0.0); break;
        case 2:
        m = vec3(0.2 + 0.4 * mod(floor(p.x) + floor(p.z), 2.0)); break;
    }
    return m;
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

vec3 getNormal(vec3 p) {
    vec2 e = vec2(EPSILON, 0.0);
    vec3 n = vec3(map(p).x) - vec3(map(p - e.xyy).x, map(p - e.yxy).x, map(p - e.yyx).x);
    return normalize(n);
}

vec3 getLight(vec3 p, vec3 rd, vec3 color) {
    vec3 lightPos = vec3(20.0, 40.0, -30.0);
    vec3 L = normalize(lightPos - p);
    vec3 N = getNormal(p);
    vec3 V = -rd;
    vec3 R = reflect(-L, N);

    vec3 specColor = vec3(0.5);
    vec3 specular = specColor * pow(clamp(dot(R, V), 0.0, 1.0), 10.0);
    vec3 diffuse = color * clamp(dot(L, N), 0.0, 1.0);
    vec3 ambient = color * 0.05;


    float d = rayMarch(p + N * 0.02, normalize(lightPos)).x;
    if (d < length(lightPos - p)) return ambient;

    return diffuse + ambient + specular;
}

mat3 getCam(vec3 ro, vec3 lookAt) {
    vec3 camF = normalize(vec3(lookAt - ro));
    vec3 camR = normalize(cross(vec3(0, 1, 0), camF));
    vec3 camU = cross(camF, camR);
    return mat3(camR, camU, camF);
}

void render(inout vec3 col, in vec2 uv) {

    vec3 ro = ubo.position;
    vec3 rd = getCam(ro, ro + ubo.direction) * normalize(vec3(uv, ubo.fov));

    vec2 object = rayMarch(ro, rd);

    vec3 background = vec3(0.5, 0.8, 0.9);
    if (object.x < MAX_DIST) {
        vec3 p = ro + object.x * rd;
        vec3 material = getMaterial(p, object.y);
        col += getLight(p, rd, material);

        col = mix(col, background, 1.0 - exp(-0.0008 * object.x * object.x));
    }else{
        col += background - max(0.95 * rd.y, 0.0);
    }


}

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = (2*gl_FragCoord.xy-vec2(ubo.width, ubo.height))/vec2(ubo.width, ubo.height)/2;
    uv.y = -uv.y;

    vec3 col = vec3(0);
    render(col, uv);


    outColor = vec4(col, 1.0);
}