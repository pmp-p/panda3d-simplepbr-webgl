shaders={
    'post.vert': '''#version 120

uniform mat4 p3d_ModelViewProjectionMatrix;

attribute vec4 p3d_Vertex;
attribute vec2 p3d_MultiTexCoord0;

varying vec2 v_texcoord;

void main() {
    v_texcoord = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
''',
    'skybox.vert': '''#version 120

uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ViewMatrix;

attribute vec4 p3d_Vertex;

varying vec3 v_texcoord;

void main() {
    v_texcoord = p3d_Vertex.xyz;
    mat4 view = mat4(mat3(p3d_ViewMatrix));
    gl_Position = p3d_ProjectionMatrix * view * p3d_Vertex;
}
''',
    'simplepbr.vert': '''#version 120


#ifndef MAX_LIGHTS
    #define MAX_LIGHTS 8
#endif

#ifdef ENABLE_SHADOWS
uniform struct p3d_LightSourceParameters {
    vec4 position;
    vec4 diffuse;
    vec4 specular;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[MAX_LIGHTS];
#endif

#ifdef ENABLE_SKINNING
uniform mat4 p3d_TransformTable[100];
#endif

uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;
uniform mat4 p3d_TextureMatrix;
uniform mat4 p3d_ModelMatrixInverseTranspose;

attribute vec4 p3d_Vertex;
attribute vec4 p3d_Color;
attribute vec3 p3d_Normal;
attribute vec4 p3d_Tangent;
attribute vec2 p3d_MultiTexCoord0;
#ifdef ENABLE_SKINNING
attribute vec4 transform_weight;
attribute vec4 transform_index;
#endif


varying vec3 v_position;
varying vec4 v_color;
varying mat3 v_tbn;
varying vec2 v_texcoord;
varying mat3 v_world_tbn;
#ifdef ENABLE_SHADOWS
varying vec4 v_shadow_pos[MAX_LIGHTS];
#endif

void main() {
#ifdef ENABLE_SKINNING
    mat4 skin_matrix = (
        p3d_TransformTable[int(transform_index.x)] * transform_weight.x +
        p3d_TransformTable[int(transform_index.y)] * transform_weight.y +
        p3d_TransformTable[int(transform_index.z)] * transform_weight.z +
        p3d_TransformTable[int(transform_index.w)] * transform_weight.w
    );
    vec4 vert_pos4 = p3d_ModelViewMatrix * skin_matrix * p3d_Vertex;
    vec4 model_normal = skin_matrix * vec4(p3d_Normal, 0.0);
#else
    vec4 vert_pos4 = p3d_ModelViewMatrix * p3d_Vertex;
    vec4 model_normal = vec4(p3d_Normal, 0.0);
#endif
    v_position = vec3(vert_pos4);
    v_color = p3d_Color;
    v_texcoord = (p3d_TextureMatrix * vec4(p3d_MultiTexCoord0, 0, 1)).xy;
#ifdef ENABLE_SHADOWS
    for (int i = 0; i < p3d_LightSource.length(); ++i) {
        v_shadow_pos[i] = p3d_LightSource[i].shadowViewMatrix * vert_pos4;
    }
#endif

    vec3 normal = normalize(p3d_NormalMatrix * model_normal.xyz);
    vec3 tangent = normalize(vec3(p3d_ModelViewMatrix * vec4(p3d_Tangent.xyz, 0.0)));
    vec3 bitangent = cross(normal, tangent) * p3d_Tangent.w;
    v_tbn = mat3(
        tangent,
        bitangent,
        normal
    );

    vec3 world_normal = normalize(p3d_ModelMatrixInverseTranspose * model_normal).xyz;
    vec3 world_tangent = normalize(vec3(p3d_ModelMatrix * vec4(p3d_Tangent.xyz, 0.0)));
    vec3 world_bitangent = cross(world_normal, world_tangent) * p3d_Tangent.w;
    v_world_tbn = mat3(
            world_tangent,
            world_bitangent,
            world_normal
    );

    gl_Position = p3d_ProjectionMatrix * vert_pos4;
}
''',
    'skybox.frag': '''#version 120


#ifdef USE_330
    #define textureCube texture

    out vec4 o_color;
#endif

uniform samplerCube skybox;

varying vec3 v_texcoord;

void main() {
    vec4 color = textureCube(skybox, v_texcoord);
#ifdef USE_330
    o_color = color;
#else
    gl_FragColor = color;
#endif
}
''',
    'shadow.vert': '''#version 120


uniform mat4 p3d_ModelViewProjectionMatrix;
#ifdef ENABLE_SKINNING
uniform mat4 p3d_TransformTable[100];
#endif

attribute vec4 p3d_Vertex;
attribute vec4 p3d_Color;
attribute vec2 p3d_MultiTexCoord0;
#ifdef ENABLE_SKINNING
attribute vec4 transform_weight;
attribute vec4 transform_index;
#endif


varying vec4 v_color;
varying vec2 v_texcoord;

void main() {
#ifdef ENABLE_SKINNING
    mat4 skin_matrix = (
        p3d_TransformTable[int(transform_index.x)] * transform_weight.x +
        p3d_TransformTable[int(transform_index.y)] * transform_weight.y +
        p3d_TransformTable[int(transform_index.z)] * transform_weight.z +
        p3d_TransformTable[int(transform_index.w)] * transform_weight.w
    );
    vec4 vert_pos4 = skin_matrix * p3d_Vertex;
#else
    vec4 vert_pos4 = p3d_Vertex;
#endif
    v_color = p3d_Color;
    v_texcoord = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * vert_pos4;
}
''',
    'simplepbr.frag': '''#version 120


// Based on code from https://github.com/KhronosGroup/glTF-Sample-Viewer


#ifndef MAX_LIGHTS
    #define MAX_LIGHTS 8
#endif

#ifdef USE_330
    #define texture2D texture
    #define textureCube texture
    #define textureCubeLod textureLod
#else
    #extension GL_ARB_shader_texture_lod : require
#endif

uniform struct p3d_MaterialParameters {
    vec4 baseColor;
    vec4 emission;
    float roughness;
    float metallic;
} p3d_Material;

uniform struct p3d_LightSourceParameters {
    vec4 position;
    vec4 diffuse;
    vec4 specular;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
#ifdef ENABLE_SHADOWS
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
#endif
} p3d_LightSource[MAX_LIGHTS];

uniform struct p3d_LightModelParameters {
    vec4 ambient;
} p3d_LightModel;

#ifdef ENABLE_FOG
uniform struct p3d_FogParameters {
    vec4 color;
    float density;
} p3d_Fog;
#endif

uniform vec4 p3d_ColorScale;
uniform vec4 p3d_TexAlphaOnly;

uniform vec3 sh_coeffs[9];

struct FunctionParamters {
    float n_dot_l;
    float n_dot_v;
    float n_dot_h;
    float l_dot_h;
    float v_dot_h;
    float roughness;
    float metallic;
    vec3 reflection0;
    vec3 diffuse_color;
    vec3 specular_color;
};

uniform sampler2D p3d_TextureBaseColor;
uniform sampler2D p3d_TextureMetalRoughness;
uniform sampler2D p3d_TextureNormal;
uniform sampler2D p3d_TextureEmission;

uniform sampler2D brdf_lut;
uniform samplerCube filtered_env_map;
uniform float max_reflection_lod;

const vec3 F0 = vec3(0.04);
const float PI = 3.141592653589793;
const float SPOTSMOOTH = 0.001;
const float LIGHT_CUTOFF = 0.001;

varying vec3 v_position;
varying vec4 v_color;
varying vec2 v_texcoord;
varying mat3 v_tbn;
varying mat3 v_world_tbn;
#ifdef ENABLE_SHADOWS
varying vec4 v_shadow_pos[MAX_LIGHTS];
#endif

#ifdef USE_330
out vec4 o_color;
#endif


// Schlick's Fresnel approximation with Spherical Gaussian approximation to replace the power
vec3 specular_reflection(FunctionParamters func_params) {
    vec3 f0 = func_params.reflection0;
    float v_dot_h= func_params.v_dot_h;
    return f0 + (vec3(1.0) - f0) * pow(2.0, (-5.55473 * v_dot_h - 6.98316) * v_dot_h);
}

vec3 fresnelSchlickRoughness(float u, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - u, 0.0, 1.0), 5.0);
}

// Smith GGX with optional fast sqrt approximation (see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg))
float visibility_occlusion(FunctionParamters func_params) {
    float r = func_params.roughness;
    float n_dot_l = func_params.n_dot_l;
    float n_dot_v = func_params.n_dot_v;
#ifdef SMITH_SQRT_APPROX
    float ggxv = n_dot_l * (n_dot_v * (1.0 - r) + r);
    float ggxl = n_dot_v * (n_dot_l * (1.0 - r) + r);
#else
    float r2 = r * r;
    float ggxv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - r2) + r2);
    float ggxl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - r2) + r2);
#endif

    float ggx = ggxv + ggxl;
    if (ggx > 0.0) {
        return 0.5 / ggx;
    }
    return 0.0;
}

// GGX/Trowbridge-Reitz
float microfacet_distribution(FunctionParamters func_params) {
    float roughness2 = func_params.roughness * func_params.roughness;
    float f = (func_params.n_dot_h * func_params.n_dot_h) * (roughness2 - 1.0) + 1.0;
    return roughness2 / (PI * f * f);
}

// Lambert
float diffuse_function() {
    return 1.0 / PI;
}

vec3 irradiance_from_sh(vec3 normal) {
    return
        + sh_coeffs[0] * 0.282095
        + sh_coeffs[1] * 0.488603 * normal.x
        + sh_coeffs[2] * 0.488603 * normal.z
        + sh_coeffs[3] * 0.488603 * normal.y
        + sh_coeffs[4] * 1.092548 * normal.x * normal.z
        + sh_coeffs[5] * 1.092548 * normal.y * normal.z
        + sh_coeffs[6] * 1.092548 * normal.y * normal.x
        + sh_coeffs[7] * (0.946176 * normal.z * normal.z - 0.315392)
        + sh_coeffs[8] * 0.546274 * (normal.x * normal.x - normal.y * normal.y);
}

void main() {
    vec4 metal_rough = texture2D(p3d_TextureMetalRoughness, v_texcoord);
    float metallic = clamp(p3d_Material.metallic * metal_rough.b, 0.0, 1.0);
    float perceptual_roughness = clamp(p3d_Material.roughness * metal_rough.g,  0.0, 1.0);
    float alpha_roughness = perceptual_roughness * perceptual_roughness;
    vec4 base_color = p3d_Material.baseColor * v_color * p3d_ColorScale * (texture2D(p3d_TextureBaseColor, v_texcoord) + p3d_TexAlphaOnly);
    vec3 diffuse_color = (base_color.rgb * (vec3(1.0) - F0)) * (1.0 - metallic);
    vec3 spec_color = mix(F0, base_color.rgb, metallic);
#ifdef USE_NORMAL_MAP
    vec3 normalmap = 2.0 * texture2D(p3d_TextureNormal, v_texcoord).rgb - 1.0;
    vec3 n = normalize(v_tbn * normalmap);
    vec3 world_normal = normalize(v_world_tbn * normalmap);
#else
    vec3 n = normalize(v_tbn[2]);
    vec3 world_normal = normalize(v_world_tbn[2]);
#endif
    vec3 v = normalize(-v_position);
    vec3 r = reflect(-v, n);

#ifdef USE_OCCLUSION_MAP
    float ambient_occlusion = metal_rough.r;
#else
    float ambient_occlusion = 1.0;
#endif

#ifdef USE_EMISSION_MAP
    vec3 emission = p3d_Material.emission.rgb * texture2D(p3d_TextureEmission, v_texcoord).rgb;
#else
    vec3 emission = vec3(0.0);
#endif

    vec4 color = vec4(vec3(0.0), base_color.a);

    float n_dot_v = clamp(abs(dot(n, v)), 0.0, 1.0);

    for (int i = 0; i < p3d_LightSource.length(); ++i) {
        vec3 lightcol = p3d_LightSource[i].diffuse.rgb;

        if (dot(lightcol, lightcol) < LIGHT_CUTOFF) {
            continue;
        }

        vec3 light_pos = p3d_LightSource[i].position.xyz - v_position * p3d_LightSource[i].position.w;
        vec3 l = normalize(light_pos);
        vec3 h = normalize(l + v);
        float dist = length(light_pos);
        vec3 att_const = p3d_LightSource[i].attenuation;
        float attenuation_factor = 1.0 / (att_const.x + att_const.y * dist + att_const.z * dist * dist);
        float spotcos = dot(normalize(p3d_LightSource[i].spotDirection), -l);
        float spotcutoff = p3d_LightSource[i].spotCosCutoff;
        float shadowSpot = smoothstep(spotcutoff-SPOTSMOOTH, spotcutoff+SPOTSMOOTH, spotcos);
#ifdef ENABLE_SHADOWS
#ifdef USE_330
        float shadowCaster = textureProj(p3d_LightSource[i].shadowMap, v_shadow_pos[i]);
#else
        float shadowCaster = shadow2DProj(p3d_LightSource[i].shadowMap, v_shadow_pos[i]).r;
#endif
#else
        float shadowCaster = 1.0;
#endif
        float shadow = shadowSpot * shadowCaster * attenuation_factor;

        FunctionParamters func_params;
        func_params.n_dot_l = clamp(dot(n, l), 0.0, 1.0);
        func_params.n_dot_v = n_dot_v;
        func_params.n_dot_h = clamp(dot(n, h), 0.0, 1.0);
        func_params.l_dot_h = clamp(dot(l, h), 0.0, 1.0);
        func_params.v_dot_h = clamp(dot(v, h), 0.0, 1.0);
        func_params.roughness = alpha_roughness;
        func_params.metallic =  metallic;
        func_params.reflection0 = spec_color;
        func_params.diffuse_color = diffuse_color;
        func_params.specular_color = spec_color;

        vec3 F = specular_reflection(func_params);
        float V = visibility_occlusion(func_params); // V = G / (4 * n_dot_l * n_dot_v)
        float D = microfacet_distribution(func_params);

        vec3 diffuse_contrib = diffuse_color * diffuse_function();
        vec3 spec_contrib = vec3(F * V * D);
        color.rgb += func_params.n_dot_l * lightcol * (diffuse_contrib + spec_contrib) * shadow;
    }


    // Indirect diffuse + specular (IBL)
    vec3 ibl_f = fresnelSchlickRoughness(n_dot_v, spec_color, perceptual_roughness);
    vec3 ibl_kd = (1.0 - ibl_f) * (1.0 - metallic);
    vec3 ibl_diff = base_color.rgb * max(irradiance_from_sh(world_normal), 0.0) * diffuse_function();

    vec2 env_brdf = texture2D(brdf_lut, vec2(n_dot_v, perceptual_roughness)).rg;
    vec3 ibl_spec_color = textureCubeLod(filtered_env_map, r, perceptual_roughness * max_reflection_lod).rgb;
    vec3 ibl_spec = ibl_spec_color * (ibl_f * env_brdf.x + env_brdf.y);
    color.rgb += (ibl_kd * ibl_diff  + ibl_spec) * ambient_occlusion;

    // Indirect diffuse (ambient light)
    color.rgb += (diffuse_color + spec_color) * p3d_LightModel.ambient.rgb * ambient_occlusion;

    // Emission
    color.rgb += emission;

#ifdef ENABLE_FOG
    // Exponential fog
    float fog_distance = length(v_position);
    float fog_factor = clamp(1.0 / exp(fog_distance * p3d_Fog.density), 0.0, 1.0);
    color = mix(p3d_Fog.color, color, fog_factor);
#endif

#ifdef USE_330
    o_color = color;
#else
    gl_FragColor = color;
#endif
}
''',
    'tonemap.frag': '''#version 120


#ifdef USE_330
    #define texture2D texture
#endif

#ifdef USE_330
    #define texture2D texture
    #define texture3D texture
#endif

uniform sampler2D tex;
#ifdef USE_SDR_LUT
    uniform sampler3D sdr_lut;
    uniform float sdr_lut_factor;
#endif
uniform float exposure;

varying vec2 v_texcoord;

#ifdef USE_330
out vec4 o_color;
#endif

void main() {
    vec3 color = texture2D(tex, v_texcoord).rgb;

    color *= exposure;
    color = max(vec3(0.0), color - vec3(0.004));
    color = (color * (vec3(6.2) * color + vec3(0.5))) / (color * (vec3(6.2) * color + vec3(1.7)) + vec3(0.06));

#ifdef USE_SDR_LUT
    vec3 lut_size = vec3(textureSize(sdr_lut, 0));
    vec3 lut_uvw = (color.rgb * float(lut_size - 1.0) + 0.5) / lut_size;
    vec3 lut_color = texture3D(sdr_lut, lut_uvw).rgb;
    color = mix(color, lut_color, sdr_lut_factor);
#endif
#ifdef USE_330
    o_color = vec4(color, 1.0);
#else
    gl_FragColor = vec4(color, 1.0);
#endif
}
''',
    'shadow.frag': '''#version 120


#ifdef USE_330
    #define texture2D texture
#endif

uniform struct p3d_MaterialParameters {
    vec4 baseColor;
} p3d_Material;

uniform vec4 p3d_ColorScale;

uniform sampler2D p3d_TextureBaseColor;
varying vec4 v_color;
varying vec2 v_texcoord;

#ifdef USE_330
out vec4 o_color;
#endif

void main() {
    vec4 base_color = p3d_Material.baseColor * v_color * p3d_ColorScale * texture2D(p3d_TextureBaseColor, v_texcoord);
#ifdef USE_330
    o_color = base_color;
#else
    gl_FragColor = base_color;
#endif
}
'''}
