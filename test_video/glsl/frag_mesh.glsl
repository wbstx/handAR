#version 330 core
#define MAX_MATERIALS 10
#extension GL_ARB_conservative_depth: enable

// Varying
// ------------------------------------
in vec4 v_color;

out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

const float c_zero = 0.0;
const int c_sze = 5;

uniform sampler2D u_texture;
uniform bool has_texture;
uniform vec3 u_materials[66];
uniform float u_time;
uniform float u_mat_rendering;
uniform mat4 u_light_mat;

varying vec2 v_texcoord;
flat in float v_mtl;
flat in float v_group;

struct lightSource
{
    vec4 position;
    vec4 diffuse;
    vec4 specular;
    float constantAttenuation, linearAttenuation, quadraticAttenuation;
    float spotCutoff, spotExponent;
    vec3 spotDirection;
};

const int numberOfLights = 2;
lightSource lights[numberOfLights];

lightSource light0 = lightSource(
vec4(0.0, 3.0, 3.0, 0.0),
vec4(0.5, 0.5, 0.5, 0.0),
vec4(1.0, 1.0, 1.0, 0.0),
0.0, 1.0, 0.0,
180.0, 0.0,
vec3(0.0, 0.0, 0.0)
);

lightSource light1 = lightSource(
vec4(0.0, 0.0, 1.0, 0.0),
vec4(1.0, 1.0, 1.0, 0.0),
vec4(0.1, 0.1, 0.1, 0.0),
0.0, 1.0, 0.0,
80.0, 10.0,
vec3(0.0, 0.0, 0.0)
);

vec4 scene_ambient = vec4(0.4, 0.4, 0.4, 1.0);

layout (depth_greater) out float gl_FragDepth;

//Point Light
void main()
{
    // if (int(u_mat_rendering) == 0)
    // if (int(v_mtl) == 0)
    // if (int(v_group) == 0)
    if (false)
    {
        if (has_texture)
        {
            vec3 objectColor = vec3(0.5, 0.5, 0.5);
            // ambient
            float ambientStrength = 1.0;
            vec3 lightColor = vec3(0.8, 0.8, 0.8);
            vec3 ambient = ambientStrength * lightColor;// lightColor;
            vec3 LightPos = vec3(0.2, 0.0, 0.0);
            // diffuse
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(LightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            // specular
            float specularStrength = 0.5;
            vec3 viewDir = normalize(-FragPos);// the viewer is always at (0,0,0) in view-space, so viewDir is (0,0,0) - Position => -Position
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;

            // texture
            vec2 new_coord = vec2(0, 0);
            new_coord[0] = v_texcoord[0];
            new_coord[1] = v_texcoord[1];
            // new_coord[0] = (new_coord[0] + 0.2) / 1.5;
            // new_coord[1] = (new_coord[1] + 0.2) / 1.5;

            vec4 tex = texture2D(u_texture, new_coord);

            // vec3 result = (ambient + diffuse + specular) * tex.rgb;
            vec3 result = (ambient + diffuse + 0.2 * specular) * tex.rgb;
            FragColor = vec4(result, 1.0);
            // FragColor = vec4(0.0, 1.0, 0.0, 1.0);
            gl_FragDepth = gl_FragCoord.z;// Z test hacking
            // gl_FragDepth = gl_FragCoord.z + 0.01;// Z test hacking
        }
        else
        {
            // ambient
            vec3 ambient_mat = u_materials[int(v_mtl)*3];
            vec3 lightColor = vec3(0.8, 0.8, 0.8);
            vec3 ambient = ambient_mat * lightColor;// lightColor;
            vec3 LightPos = vec3(0.0, 3.0, 3.0);
            // diffuse
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(LightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor * u_materials[int(v_mtl)*3 + 1];

            // specular
            // vec3 specular_mat = u_materials[int(v_group) + 2];
            vec3 specular_mat = u_materials[int(v_mtl)*3 + 2];
            vec3 viewDir = normalize(-FragPos);// the viewer is always at (0,0,0) in view-space, so viewDir is (0,0,0) - Position => -Position
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specular_mat * spec * lightColor;

            vec3 result = (ambient + diffuse + specular);
            FragColor = vec4(result, 1.0);
            gl_FragDepth = gl_FragCoord.z;
            // gl_FragDepth = gl_FragCoord.z + 0.01;// Z test hacking
            // gl_FragDepth = gl_FragCoord.z + 0.004;// Z test hacking
        }
    }
    else
    {
        if (has_texture)
        {
            vec3 objectColor = vec3(0.5, 0.5, 0.5);
            // ambient
            float ambientStrength = 1.0;
            vec3 lightColor = vec3(0.8, 0.8, 0.8);
            vec3 ambient = ambientStrength * lightColor;// lightColor;
            vec3 LightPos = vec3(0.2, 0.0, 0.0);
            // diffuse
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(LightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            // specular
            float specularStrength = 0.5;
            vec3 viewDir = normalize(-FragPos);// the viewer is always at (0,0,0) in view-space, so viewDir is (0,0,0) - Position => -Position
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;

            // texture
            vec2 new_coord = vec2(0, 0);
            new_coord[0] = v_texcoord[0];
            new_coord[1] = v_texcoord[1];
            // new_coord[0] = (new_coord[0] + 0.2) / 1.5;
            // new_coord[1] = (new_coord[1] + 0.2) / 1.5;

            vec4 tex = texture2D(u_texture, new_coord);

            // vec3 result = (ambient + diffuse + specular) * tex.rgb;
            vec3 result = (ambient + diffuse + 0.2 * specular) * tex.rgb;
            FragColor = vec4(result, 1.0);
            gl_FragDepth = gl_FragCoord.z;// Z test hacking
            // gl_FragDepth = gl_FragCoord.z + 0.01;// Z test hacking
        }
        else
        {
            //            // ambient
            //            vec3 ambient_mat = u_materials[int(v_mtl)*3];
            //            vec3 lightColor = vec3(0.8, 0.8, 0.8);
            //            vec3 ambient = ambient_mat * lightColor;// lightColor;
            //            vec3 LightPos = vec3(0.0, 3.0, 3.0);
            //            // diffuse
            //            vec3 norm = normalize(Normal);
            //            vec3 lightDir = normalize(LightPos - FragPos);
            //            float diff = max(dot(norm, lightDir), 0.0);
            //            vec3 diffuse = diff * lightColor * u_materials[int(v_mtl)*3 + 1];
            //
            //            // specular
            //            // vec3 specular_mat = u_materials[int(v_group) + 2];
            //            vec3 specular_mat = u_materials[int(v_mtl)*3 + 2];
            //            vec3 viewDir = normalize(-FragPos);// the viewer is always at (0,0,0) in view-space, so viewDir is (0,0,0) - Position => -Position
            //            vec3 reflectDir = reflect(-lightDir, norm);
            //            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            //            vec3 specular = specular_mat * spec * lightColor;
            //
            //            vec3 result = (ambient + diffuse + specular);
            //            FragColor = vec4(result, 1.0);
            //            gl_FragDepth = gl_FragCoord.z;
            //            // gl_FragDepth = gl_FragCoord.z + 0.01;// Z test hacking
            //}

            lights[0] = light0;
            lights[1] = light1;
            vec3 normalDirection = normalize(Normal);
            vec3 viewDirection = normalize(-FragPos);
            vec3 lightDirection;
            float attenuation;

            // initialize total lighting with ambient lighting
            vec3 ambient_mat = u_materials[int(v_mtl)*3];
            vec3 totalLighting = vec3(scene_ambient) * ambient_mat;

            for (int index = 0; index < numberOfLights; index++)// for all light sources
            {
                vec4 LightPos = u_light_mat * lights[index].position;
                // vec4 LightPos = lights[index].position;
                if (0.0 == lights[index].position.w)// directional light?
                {
                    attenuation = 1.0;// no attenuation
                    lightDirection = normalize(vec3(LightPos));
                }
                else // point light or spotlight (or other kind of light)
                {
                    vec3 positionToLightSource = vec3(LightPos - vec4(FragPos, 1.0));
                    float distance = length(positionToLightSource);
                    lightDirection = normalize(positionToLightSource);
                    attenuation = 1.0 / (lights[index].constantAttenuation
                    + lights[index].linearAttenuation * distance
                    + lights[index].quadraticAttenuation * distance * distance);

                    if (lights[index].spotCutoff <= 90.0)// spotlight?
                    {
                        float clampedCosine = max(0.0, dot(-lightDirection, normalize(lights[index].spotDirection)));
                        if (clampedCosine < cos(radians(lights[index].spotCutoff)))// outside of spotlight cone?
                        {
                            attenuation = 0.0;
                        }
                        else
                        {
                            attenuation = attenuation * pow(clampedCosine, lights[index].spotExponent);
                        }
                    }
                }

                vec3 diffuse_mat = u_materials[int(v_mtl)*3 + 1];
                vec3 diffuseReflection = attenuation
                * vec3(lights[index].diffuse) * diffuse_mat
                * max(0.0, dot(normalDirection, lightDirection));

                vec3 specularReflection;
                if (dot(normalDirection, lightDirection) < 0.0)// light source on the wrong side?
                {
                    specularReflection = vec3(0.0, 0.0, 0.0);// no specular reflection
                }
                else // light source on the right side
                {
                    vec3 specular_mat = u_materials[int(v_mtl)*3 + 2];
                    specularReflection = attenuation * vec3(lights[index].specular) * specular_mat
                    * pow(max(0.0, dot(reflect(-lightDirection, normalDirection), viewDirection)), 5.0);
                }

                totalLighting = totalLighting + diffuseReflection + specularReflection;
            }

            FragColor = vec4(totalLighting, 1.0);
            gl_FragDepth = gl_FragCoord.z;
        }
    }
}