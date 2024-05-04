#version 450
//#extension GL_GOOGLE_include_directive : enable
#include "hg_sdf.glsl"

layout (set = 0, binding = 0) uniform ParameterUBO {
    vec3 position;
    vec3 direction;
    float frame;
    float fov;
    int width;
    int height;
} ubo;

const int MAX_STEPS = 256;
const float MAX_DIST = 500;
const float EPSILON = 0.001;

float fDisplace(vec3 p){
    float time = ubo.frame/100;
    pR(p.yz, sin(2.0 * time));
    return (sin(p.x + 4.0 * time) * sin(p.y + sin(2.0 * time)) * sin(p.z + 6.0 * time));
}

vec2 fOpUnionID(vec2 res1, vec2 res2) {
    return (res1.x < res2.x) ? res1 : res2;
}

vec2 fOpDifferenceID(vec2 res1, vec2 res2) {
    return (res1.x > - res2.x) ? res1 : vec2(-res2.x, res2.y);
}

vec2 fOpDifferenceColumnsID(vec2 res1, vec2 res2, float r, float n){
    float dist = fOpDifferenceColumns(res1.x, res2.x, r, n);
    return (res1.x > -res2.x) ? vec2(dist, res1.y) : vec2(dist, res2.y);
}

vec2 fOpUnionStairsID(vec2 res1, vec2 res2, float r, float n){
    float dist = fOpUnionStairs(res1.x, res2.x, r, n);
    return (res1.x < res2.x) ? vec2(dist, res1.y) : vec2(dist, res2.y);
}

vec2 fOpUnionChamferID(vec2 res1, vec2 res2, float r){
    float dist = fOpUnionChamfer(res1.x, res2.x, r);
    return (res1.x < res2.x) ? vec2(dist, res1.y) : vec2(dist, res2.y);
}

vec2 map(vec3 p) {
    vec3 tmp, op = p;
    
    float planeDist = fPlane(p, vec3(0, 1, 0), 14.0);
    float planeID = 2.0;
    vec2 plane = vec2(planeDist, planeID);

    vec3 pb = p;
    float cubeDist = fBoxCheap(pb, vec3(6));
    float cubeID = 1.0;
    vec2 cube = vec2(cubeDist, cubeID);

//    float sphereDist = fSphere(p, 9 + fDisplace(p));
//    float sphereID = 1.0;
//    vec2 sphere = vec2(sphereDist, sphereID);

    pMirrorOctant(p.xz, vec2(50, 50));
    p.x = -abs(p.x) + 20;
    pMod1(p.z, 15);

    vec3 pr = p;
    pr.y -= 15.7;
    pR(pr.xy, 0.6);
    pr.x -= 16.0;
    float roofDist = fBox2Cheap(pr.xy, vec2(20, 0.3));
    float roofID = 4.0;
    vec2 roof = vec2(roofDist, roofID);

    float boxDist = fBoxCheap(p, vec3(3, 9, 4));
    float boxID = 3.0;
    vec2 box = vec2(boxDist, boxID);

    vec3 pc = p;
    pc.y -= 9.0;
    float cylinderDist = fCylinder(pc.yxz, 4, 3);
    float cylinderID = 3.0;
    vec2 cylinder = vec2(cylinderDist, cylinderID);

    float wallDist = fBox2Cheap(p.xy, vec2(1, 15));
    float wallID = 3.0;
    vec2 wall = vec2(wallDist, wallID);

    vec2 res;
    //res = wall;
    res = fOpUnionID(cylinder, box);
    res = fOpDifferenceColumnsID(wall, res, 0.6, 3.0);
    res = fOpUnionChamferID(res, roof, 0.9);
    res = fOpUnionStairsID(res, plane, 4.0, 5.0);
    //res = fOpUnionID(res, sphere);
    res = fOpUnionID(res, cube);
    return res;
}

vec3 getMaterial(vec3 p, float id, vec3 normal) { 
    vec3 m;
    switch (int(id)) {
        case 1:
        m = vec3(0.9, 0.0, 0.0); break;

        case 2:
        m = vec3(0.2 + 0.4 * mod(floor(p.x) + floor(p.z), 2.0)); break;

        case 3:
        m = vec3(0.7, 0.8, 0.9); break;

        case 4: 
        vec2 i = step(fract(0.5 * p.xz), vec2(1.0 / 10.0));
        m = ((1.0 - i.x) * (1.0 - i.y)) * vec3(0.37, 0.12 ,0.0); break;

        case 5:
        break;

        default:
        m = vec3(0.4); break;
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
        if (abs(hit.x) < EPSILON || object.x > MAX_DIST ) break;
    
    }
    return object;
}

vec3 getNormal(vec3 p) {
    vec2 e = vec2(EPSILON, 0.0);
    vec3 n = vec3(map(p).x) - vec3(map(p - e.xyy).x, map(p - e.yxy).x, map(p - e.yyx).x);
    return normalize(n);
}

float getSoftShadow(vec3 p, vec3 lightPos) {
    float res = 1.0;
    float dist = 0.01;
    float lightSize = 0.03;
    for (int i = 0; i < MAX_STEPS; i++){
        float hit = map(p + lightPos * dist).x;
        res = min(res, hit / (dist * lightSize));
        dist += hit;
        if (hit < 0.0001 || dist > 60.0) break;
    }
    return clamp(res, 0.0, 1.0);
}



vec3 getLight(vec3 p, vec3 rd, float id) {
    vec3 lightPos = vec3(20.0, 40.0, -30.0);
    vec3 L = normalize(lightPos - p);
    vec3 N = getNormal(p);
    vec3 V = -rd;
    vec3 R = reflect(-L, N);

    vec3 color = getMaterial(p, id, N);

    vec3 specColor = vec3(0.5);
    vec3 specular = specColor * pow(clamp(dot(R, V), 0.0, 1.0), 10.0);
    vec3 diffuse = color * clamp(dot(L, N), 0.0, 1.0);
    vec3 ambient = color * 0.05;
    vec3 fresnel = 0.25 * color * pow(1.0 + dot(rd, N), 3.0);


    float shadow = getSoftShadow(p + N * 0.02, normalize(lightPos));

    return fresnel + ambient + (specular + diffuse) * shadow;
}

mat3 getCam(vec3 ro, vec3 lookAt) {
    vec3 camF = normalize(vec3(lookAt - ro));
    vec3 camR = normalize(cross(vec3(0, 1, 0), camF));
    vec3 camU = cross(camF, camR);
    return mat3(camR, camU, camF);
}

vec3 render(vec2 uv) {
    vec3 col = vec3(0);

    vec3 ro = ubo.position;
    vec3 rd = getCam(ro, ro + ubo.direction) * normalize(vec3(uv, ubo.fov));

    vec2 object = rayMarch(ro, rd);

    vec3 background = vec3(0.5, 0.8, 0.9);
    if (object.x < MAX_DIST) {
        vec3 p = ro + object.x * rd;
        col += getLight(p, rd, object.y);

        col = mix(col, background, 1.0 - exp(-0.00008 * object.x * object.x));
    }else{
        col += background - max(0.95 * rd.y, 0.0);
    }

    return col;
}

vec2 getUV(vec2 offset){
    return (2 * (gl_FragCoord.xy + offset)-vec2(ubo.width, ubo.height))/vec2(ubo.width, ubo.height)/2 * vec2(1, -1);
}

vec3 renderAAx4(){
    vec4 e = vec4(0.125, -0.125, 0.375, -0.375);
    vec3 colAA = render(getUV(e.xz)) + render(getUV(e.yw)) + render(getUV(e.wx)) + render(getUV(e.zy));
    return colAA / 4;
}

layout(location = 0) out vec4 outColor;

void main() {

    vec3 col = renderAAx4();
    outColor = vec4(col, 1.0);
}