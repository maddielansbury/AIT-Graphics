
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <glew.h>		 
#include <freeglut.h>	
#endif

#include <vector>
#include <vec2.h>
#include <vec3.h>
#include <vec4.h>
#include <mat4x4.h>

const unsigned int windowWidth = 700, windowHeight = 700;

int majorVersion = 3, minorVersion = 0;


// image to be computed by ray tracing
vec3 image[windowWidth * windowHeight];
class LightSource
{
protected:
	vec3 powerDensity;
public:
	virtual vec3 getPowerDensityAt(vec3 x) = 0;
	virtual vec3 getLightDirAt(vec3 x) = 0;
	virtual float getDistanceFrom(vec3 x) = 0;
};

class DirectionalLight : public LightSource
{
	vec3 lightDir;

public:
	DirectionalLight(vec3 pD, vec3 dir)
	{
		powerDensity = pD;
		lightDir = dir;
	};

	vec3 getPowerDensityAt(vec3 x)
	{
		return powerDensity;
	}

	vec3 getLightDirAt(vec3 x)
	{
		return lightDir;
	}

	float getDistanceFrom(vec3 x)
	{
		return FLT_MAX;
	}
};

class PointLight : public LightSource
{
	vec3 lightPos;

public:
	PointLight(vec3 pD, vec3 pos)
	{
		powerDensity = pD;
		lightPos = pos;
	};

	vec3 getPowerDensityAt(vec3 x)
	{
		//float diff = (x - lightPos).norm2();
		//return vec3(powerDensity.x - diff, powerDensity.y - diff, powerDensity.z - diff);
		return powerDensity;
	}

	vec3 getLightDirAt(vec3 x)
	{
		return (x - lightPos).normalize();
	}

	float getDistanceFrom(vec3 x)
	{
		return (x - lightPos).norm();
	}
	//getdistancefrom use norm on distance between position and light source position
};

class Material
{
protected:
	vec3 frontColor;
	vec3 backColor;
public:
	Material(vec3 col) : frontColor(col), backColor(vec3(0, 0, 0)) {}

	virtual vec3 shade(vec3 position, vec3 normal, LightSource* lightSource) = 0;
	virtual boolean isReflective()
	{
		return false;
	};
};

class LightSourceMaterial : public Material
{
protected:
	vec3 frontColor;
public:
	LightSourceMaterial(vec3 col) : Material(col) {
		frontColor = col;
	}

	virtual vec3 shade(vec3 position, vec3 normal, LightSource* lightSource)
	{
		return frontColor;
	}
	virtual boolean isReflective()
	{
		return false;
	};
};

class DiffuseMaterial : public Material
{
public:
	DiffuseMaterial(vec3 col) : Material(col) {}

	virtual vec3 shade(vec3 position, vec3 normal, LightSource* lightSource)
	{
		vec3 M = lightSource->getPowerDensityAt(position);
		float dotVal = max(0.0f,normal.dot(lightSource->getLightDirAt(position)));
		return M * frontColor * dotVal;
	}
};

class Sand : public DiffuseMaterial
{
	float scale;
	float turbulence;
	float period;
	float sharpness;
	vec3 frontColor;
public:
	Sand() :
		DiffuseMaterial(vec3(1, 1, 1))
	{
		scale = 16;
		turbulence = 30;
		period = 32;
		sharpness = 1;
	}
	float snoise(vec3 r) {
		unsigned int x = 0x0625DF73;
		unsigned int y = 0xD1B84B45;
		unsigned int z = 0x152AD8D0;
		float f = 0;
		for (int i = 0; i<32; i++) {
			vec3 s(x / (float)0xffffffff,
				y / (float)0xffffffff,
				z / (float)0xffffffff);
			f += sin(s.dot(r));
			x = x << 1 | x >> 31;
			y = y << 1 | y >> 31;
			z = z << 1 | z >> 31;
		}
		return f / 64.0 + 0.5;
	}

	virtual vec3 shade(vec3 position, vec3 normal, LightSource* lightSource)
	{
		vec3 M = lightSource->getPowerDensityAt(position);
		float w = position.z * period + pow(snoise(position * scale), sharpness) * turbulence;
		w = pow(sin(w) * 0.5 + 0.3, 4);
		vec3 color = (vec3(0.87, 0.76, 0.64) * w + vec3(0.94, 0.82, 0.72) * (1 - w));
		float dotVal = max(0.0f, normal.dot(lightSource->getLightDirAt(position)));
		return M * color * dotVal;
	}
};

class ShinyMaterial : public Material
{
	vec3 specCol;
	float shininess;
public:
	ShinyMaterial(vec3 col, float shininess) : Material(col), specCol(vec3(1, 1, 1)), shininess(shininess) {}
	ShinyMaterial(vec3 col, vec3 sCol, float shininess) : Material(col), specCol(sCol), shininess(shininess) {}

	virtual vec3 shade(vec3 position, vec3 normal, LightSource* lightSource)
	{
		vec3 M = lightSource->getPowerDensityAt(position);
		vec3 H = (position + lightSource->getLightDirAt(position)).normalize();
		float dotVal = max(0.0f, H.dot(normal));
		return M * frontColor * max(0.0f, normal.dot(lightSource->getLightDirAt(position))) + M * specCol * pow(dotVal,shininess);
	}
};

class BeachBall : public ShinyMaterial
{
	vec3 specCol;
	float shininess;
public:
	BeachBall() : ShinyMaterial(vec3(1,1,1), 20) {
		specCol = vec3(1, 1, 1);
		shininess = 20;
	}

	virtual vec3 shade(vec3 position, vec3 normal, LightSource* lightSource)
	{
		vec3 color = vec3(1,1,1);
		float theta = (acos(normal.z / normal.norm()));
		theta = (theta / M_PI) * 100;
		float phi = (atan(normal.y / normal.x));
		phi = (phi / M_PI) * 180;
		
		
		if (phi >-90.0 && phi < -45.0 && theta > 0.0 && theta < 180.0){
            color = vec3(0.75, 0, 0.93);
        }
        else if (phi >-45.0 && phi < 0.0 && theta > 0.0 && theta < 180.0){
            color = vec3(0.99, 0.46, 1);
        }
        else if (phi >0.0 && phi < 45.0 && theta > 0.0 && theta < 180.0){
            color = vec3(0.56, 0.03, 0.63);
        }
        else{
            color = vec3(0.99, 0.94, 1);
        }
		

		vec3 M = lightSource->getPowerDensityAt(position);
		vec3 H = (position + lightSource->getLightDirAt(position)).normalize();
		float dotVal = max(0.0f, H.dot(normal));
		return M * color * max(0.0f, normal.dot(lightSource->getLightDirAt(position))) + M * specCol * pow(dotVal, shininess);
	}
};


class ReflectiveMaterial : public Material
{
public:
	ReflectiveMaterial(vec3 col) : Material(col) {}

	virtual vec3 shade(vec3 position, vec3 normal, LightSource* lightSource)
	{
		vec3 M = lightSource->getPowerDensityAt(position);
		vec3 H = (position + lightSource->getLightDirAt(position)) / 2;
		float dotVal = max(0.0f, H.dot(normal));
		return M * frontColor * max(0.0f, normal.dot(lightSource->getLightDirAt(position))) + M * vec3(1, 1, 1) * pow(dotVal, 100);
	}

	virtual boolean isReflective()
	{
		return true;
	};
};

class Ocean : public ReflectiveMaterial
{
public:
	Ocean(vec3 col) : ReflectiveMaterial(col) {}

	vec3 snoiseGrad(vec3 r)
	{
		unsigned int x = 0x0625DF73;
		unsigned int y = 0xD1B84B45;
		unsigned int z = 0x152AD8D0;
		vec3 f = vec3(0, 0, 0);
		for (int i = 0; i<32; i++)
		{
			vec3 s(x / (float)0xffffffff,
				y / (float)0xffffffff,
				z / (float)0xffffffff);
			f += s * cos(s.dot(r));
			x = x << 1 | x >> 31;
			y = y << 1 | y >> 31;
			z = z << 1 | z >> 31;
		}
		return f * (1.0 / 64.0);
	}
	
	virtual vec3 shade(vec3 position, vec3 normal, LightSource* lightSource)
	{
		vec3 gradNormal = (normal + snoiseGrad(position * 8) * 5).normalize();
		vec3 M = lightSource->getPowerDensityAt(position);
		vec3 H = (position + lightSource->getLightDirAt(position)).normalize();
		float dotVal = max(0.0f, H.dot(gradNormal));
		return M * frontColor * max(0.0f, gradNormal.dot(lightSource->getLightDirAt(position))) + M * vec3(0.82, 0.96, 1) * pow(dotVal, 20);


	}
};

//make gradient function to change normal, and then IN SHADE FUNCTION add gradient

//color produced from texturing is kd


// Camera class.
class Camera
{
	vec3 eye;		//< world space camera position
	vec3 lookAt;	//< center of window in world space
	vec3 right;		//< vector from window center to window right-mid (in world space)
	vec3 up;		//< vector from window center to window top-mid (in world space)

public:
	Camera()
	{
		eye = vec3(0, 0, 5);
		lookAt = vec3(0, 0, 2);
		right = vec3(1, 0, 0);
		up = vec3(0, 1, 0);
	}
	vec3 getEye()
	{
		return eye;
	}
	// compute ray through pixel at normalized device coordinates
	vec3 rayDirFromNdc(float x, float y) {
		return (lookAt - eye
			+ right * x
			+ up    * y
			).normalize();
	}
};

// Ray structure.
class Ray
{
public:
	vec3 origin;
	vec3 dir;
	Ray(vec3 o, vec3 d)
	{
		origin = o;
		dir = d;
	}
};

// Hit record structure. Contains all data that describes a ray-object intersection point.
class Hit
{
public:
	Hit()
	{
		t = -1;
	}
	float t;				//< Ray paramter at intersection. Negative means no valid intersection.
	vec3 position;			//< Intersection coordinates.
	vec3 normal;			//< Surface normal at intersection.
	Material* material;		//< Material of intersected surface.
};

// Abstract base class.
class Intersectable
{
protected:
	Material* material;
public:
	Intersectable(Material* material) :material(material) {}
	virtual Hit intersect(const Ray& ray) = 0;
};

// Simple helper class to solve quadratic equations with the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and store the results.
class QuadraticRoots
{
public:
	float t1;
	float t2;
	// Solves the quadratic a*t*t + b*t + c = 0 using the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and sets members t1 and t2 to store the roots.
	QuadraticRoots(float a, float b, float c)
	{
		float discr = b * b - 4.0 * a * c;
		if (discr < 0) // no roots
		{
			t1 = -1;
			t2 = -1;
			return;
		}
		float sqrt_discr = sqrt(discr);
		t1 = (-b + sqrt_discr) / 2.0 / a;
		t2 = (-b - sqrt_discr) / 2.0 / a;
	}
	// Returns the lesser of the positive solutions, or a negative value if there was no positive solution.
	float getLesserPositive()
	{
		return (0 < t1 && (t2 < 0 || t1 < t2)) ? t1 : t2;
	}
};

class Plane : public Intersectable
{
	/*
	plane interssect function:
	define plane by 1 reference point & 1 normal vector

	determine if point p is on plane (implicit equation), if we know normal n and ref point r:
	subtract ref point from position p to get direction of point
	calculate dot product of normal and direction of point
	dot((p - r),n)=0 for all points on plane

	how to calculate intersection between ray (e+d.dot(t)) and plane (p-r).dot(n)?
	substitute ray for p, look for t!!! for hit record
	so d.dot(n).dot(t) = (r-e).dot(n)
	t = ((r-e).dot(n))/(d.dot(n))

	return negative value if denominator is 0

	normal vector of intersection is constant
	*/
	vec3 n;
	vec3 r0;
public:
	Plane(Material* material) : Intersectable(material), n(vec3(0, 1, 0)), r0(vec3(0, -1, 0)) {}

	vec3 getNormalAt(vec3 r)
	{
		return n;
	}
	Hit intersect(const Ray& ray)
	{
		//float t = solveQuadratic(ray).getLesserPositive();
		float t = ((r0 - ray.origin).dot(n)) / (ray.dir.dot(n));

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = getNormalAt(hit.position);

		return hit;
	}
};

class Quadric : public Intersectable
{
	mat4x4 coeffs;

public:
	Quadric(Material* material) :
		Intersectable(material)
	{
		coeffs = mat4x4(); // represents sphere of radius 1
	}

	QuadraticRoots solveQuadratic(const Ray& ray)
	{
		vec4 d = vec4(ray.dir, 0);
		vec4 e = vec4(ray.origin, 1);
		float a = d.dot(coeffs * d); //dAd.T
		float b = d.dot(coeffs * e) + e.dot(coeffs * d); //dAe.T + eAd.T
		float c = e.dot(coeffs * e); //eAe.T
		return QuadraticRoots(a, b, c);

		/*
		take implicit equatoin of quadratic surface, substitute e+dt for p in pApT = 0
		d is direction of ray, e is position/origin
		in order to extend to 4x4, for positions we add homogeneous coord 1, for direction we add homogeneous coord 0
		*/

	}

	vec3 getNormalAt(vec3 pos)
	{
		vec4 x = vec4(pos);
		vec4 result = coeffs * pos + pos * coeffs;
		return vec3(result.x, result.y, result.z).normalize();
	}

	Hit intersect(const Ray& ray)
	{
		// This is a generic intersect that works for any shape with a quadratic equation. solveQuadratic should solve the proper equation (+ ray equation) for the shape,
		// and getNormalAt should return the proper normal
		float t = solveQuadratic(ray).getLesserPositive();

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = getNormalAt(hit.position);

		return hit;
	}

	bool contains(vec3 r)
	{
		vec4 rHomo(r);
		if ((rHomo.dot(coeffs * rHomo)) < 0) return true;
		else return false;

		// evaluate implicit eq
		// return true if negative
		// return false if positive
		//substitute position to implicit equation of quadric surface
		//rhomo.dot(A*rhomo)
		//if negative, contained (true). if >= 0, not contained (False)
	}

	Quadric* transform(mat4x4 tMatrix)
	{
		coeffs = tMatrix.invert() * coeffs * tMatrix.invert().transpose();
		return this;
	}

	Quadric* sphere()
	{
		coeffs._33 = -1;
		return this;
	}

	Quadric* cylinder() //infinite along y (x^2 + z^2 - 1 = 0)
	{
		coeffs._11 = 0;
		coeffs._33 = -1;
		return this;
	}

	Quadric* cone() //infinite along y, 45deg
	{
		coeffs._11 = -1;
		coeffs._33 = 0;
		return this;
	}

	Quadric* paraboloid() //infinite along y
	{
		coeffs._11 = 0;
		coeffs._13 = -1;
		coeffs._33 = 0;
		return this;
	}

	// infinite slab, ideal for clipping
	Quadric* parallelPlanes() {
		coeffs._00 = 0;
		coeffs._11 = 1;
		coeffs._22 = 0;
		coeffs._33 = -1;
		return this;
	}

};

class ClippedQuadric : public Intersectable
{
	Quadric *shape, *clipper;

public:
	ClippedQuadric(Material* material) :
		Intersectable(material)
	{
		shape = new Quadric(material);
		clipper = new Quadric(material);
	}

	Hit intersect(const Ray& ray)
	{
		// This is a generic intersect that works for any shape with a quadratic equation.
		QuadraticRoots roots = shape->solveQuadratic(ray);
		vec3 p1 = ray.origin + ray.dir * roots.t1;
		vec3 p2 = ray.origin + ray.dir * roots.t2;

		if (!clipper->contains(p1)) roots.t1 = -1;
		if (!clipper->contains(p2)) roots.t2 = -1;

		float t = roots.getLesserPositive();

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = shape->getNormalAt(hit.position);

		return hit;
	}

	ClippedQuadric* transform(mat4x4 tMatrix)
	{
		shape = shape->transform(tMatrix);
		clipper = clipper->transform(tMatrix);
		return this;
	}

	ClippedQuadric* sphere(float radius, float height, float part)
	{
		shape = shape->sphere()->transform(mat4x4::scaling(vec3(radius, radius, radius)));
		clipper = clipper->parallelPlanes()->transform(mat4x4::scaling(vec3(1, radius * height, 1)) * mat4x4::translation(vec3(0, part, 0)));
		return this;
	}

	ClippedQuadric* cylinder(float radius, float height)
	{
		shape = shape->cylinder()->transform(mat4x4::scaling(vec3(radius, 1, radius)));
		clipper = clipper->parallelPlanes()->transform(mat4x4::scaling(vec3(1, height, 1)));
		return this;
	}

	ClippedQuadric* cone(float radius, float height)
	{
		shape = shape->cone()->transform(mat4x4::scaling(vec3(radius, 1, radius)));
		clipper = clipper->parallelPlanes()->transform(mat4x4::scaling(vec3(1, height, 1)) * mat4x4::translation(vec3(0, height, 0)));
		return this;
	}

	ClippedQuadric* paraboloid(float height)
	{
		shape = shape->paraboloid();
		return this;
	}

	//cylinder(float radius, float height)
	//cone(float height)
	//paraboloid(float height)

	//p1 = origin + dir*root.t1 ; p2 = origin + dir*root.t2
	//check if inside or outside clipping geometry
	//if one is outside, modify t to negative and call getlesserpositive on roots

	//contains quadrics shape and quadric
	//calculate intersection between ray and quadric a
	//check if intersection point is inside or outside the geometry
	//if outside, invalidate by setting t to negative
	//return getlesserpositive t

	//UMBRELLA: clip sphere so only top is visible
	//SANDCASTLE: two cylinders with cone on top, some cuboid walls (w/ procedural texturing)

};

class Scene
{
	Camera camera;
	std::vector<Intersectable*> objects;
	std::vector<Material*> materials;
	std::vector<LightSource*> lightsources;

public:
	Scene()
	{
		//water
		materials.push_back(new Ocean(vec3(0.71, 0.91, 1)));
		//sand
		materials.push_back(new Sand());
		//parasol pole
		materials.push_back(new DiffuseMaterial(vec3(0.36, 0.22, 0.16)));
		//parasol shade
		materials.push_back(new DiffuseMaterial(vec3(1, 0.31, 0.35)));
		//tree leaves
		materials.push_back(new ShinyMaterial(vec3(0.19, 0.51, 0.07), 20));
		//beach ball
		materials.push_back(new BeachBall());
		materials.push_back(new LightSourceMaterial(vec3(1, 0, 0)));

		//water
		objects.push_back(new Plane(materials[0]));
		//sand
		objects.push_back((new ClippedQuadric(materials[1]))->sphere(2, 1, -0.6)->
			transform(mat4x4::rotation(vec3(0, 0, 1), M_PI) * mat4x4::translation(vec3(0, -2.2, -1.0))));
		//parasol pole
		objects.push_back((new ClippedQuadric(materials[2]))->cylinder(0.01, 0.7)->
			transform(mat4x4::translation(vec3(-0.2, 0.03, -0.2))));
		//parasol shade
		objects.push_back((new ClippedQuadric(materials[3]))->sphere(0.9, 1, -1.6)->
			transform(mat4x4::rotation(vec3(0, 0, 1), M_PI) * mat4x4::translation(vec3(-0.2, -0.2, -0.2))));
		//tree trunk
		objects.push_back((new ClippedQuadric(materials[2]))->cone(0.2, 0.4)->
			transform(mat4x4::rotation(vec3(0, 0, 1), M_PI) * mat4x4::translation(vec3(1, 0.1, -0.6))));
		objects.push_back((new ClippedQuadric(materials[2]))->cone(0.16, 0.35)->
			transform(mat4x4::rotation(vec3(0, 0, 1), M_PI) * mat4x4::rotation(vec3(0, 0, 1), -0.25) * mat4x4::translation(vec3(1.17, 0.55, -0.6))));
		objects.push_back((new ClippedQuadric(materials[2]))->cone(0.12, 0.3)->
			transform(mat4x4::rotation(vec3(0, 0, 1), M_PI) * mat4x4::rotation(vec3(0, 0, 1), -0.32) * mat4x4::translation(vec3(1.29, 0.85, -0.6))));
		objects.push_back((new ClippedQuadric(materials[2]))->cone(0.11, 0.2)->
			transform(mat4x4::rotation(vec3(0, 0, 1), M_PI) * mat4x4::rotation(vec3(0, 0, 1), -0.42) * mat4x4::translation(vec3(1.38, 1, -0.6))));
		//tree leaves
		objects.push_back((new ClippedQuadric(materials[4]))->sphere(0.5, 1, -0.9)->
			transform(mat4x4::scaling(vec3(0.3, 1, 1.6)) *
				mat4x4::rotation(vec3(0, 0, 1), M_PI) * mat4x4::rotation(vec3(1, 0, 0), 1) *
				mat4x4::translation(vec3(1.35, 0.45, -0.6))));
		objects.push_back((new ClippedQuadric(materials[4]))->sphere(0.5, 1, -0.9)->
			transform(mat4x4::scaling(vec3(0.3, 1, 1.6)) *
				mat4x4::rotation(vec3(0, 0, 1), M_PI) * mat4x4::rotation(vec3(1, 0, 0), 0.7) * mat4x4::rotation(vec3(0, 1, 0), 1.5) *
				mat4x4::translation(vec3(1.4, 0.45, -0.55))));
		objects.push_back((new ClippedQuadric(materials[4]))->sphere(0.5, 1, -0.9)->
			transform(mat4x4::scaling(vec3(0.3, 1, 1.6)) *
				mat4x4::rotation(vec3(0, 0, 1), M_PI) * mat4x4::rotation(vec3(1, 0, 0), 0.7) * mat4x4::rotation(vec3(0, 1, 0), -1.5) *
				mat4x4::translation(vec3(1.30, 0.45, -0.55))));
		//beach ball
		objects.push_back((new ClippedQuadric(materials[5]))->sphere(0.15, 1, 0)->transform(mat4x4::translation(vec3(-1, -0.90, 0.4))));
		
		//objects.push_back((new ClippedQuadric(materials[6]))->sphere(0.025, 1, 0)->transform(mat4x4::translation(vec3(0.7, 0.4, -0.8))));

		lightsources.push_back(new DirectionalLight(vec3(0.5, 0.48, 0.47), vec3(-1, 0.8, 1.3)));
		lightsources.push_back(new DirectionalLight(vec3(0.1, 0.099, 0.094), vec3(0, 0, 1)));
		lightsources.push_back(new PointLight(vec3(0.5, 0, 0), vec3(0.7, 0.4, -0.7)));
	}
	~Scene()
	{
		for (std::vector<Material*>::iterator iMaterial = materials.begin(); iMaterial != materials.end(); ++iMaterial)
			delete *iMaterial;
		for (std::vector<Intersectable*>::iterator iObject = objects.begin(); iObject != objects.end(); ++iObject)
			delete *iObject;
	}

public:
	Camera& getCamera()
	{
		return camera;
	}

	Hit firstIntersect(const Ray& ray)
	{
		Hit minimum;
		minimum.t = 1000000;
		for (int i = 0; i < objects.size(); i++)
		{
			Hit hit = objects[i]->intersect(ray);
			if (hit.t > 0 && hit.t < minimum.t) minimum = hit;
		}
		if (minimum.t == 1000000) minimum.t *= -1;
		return minimum;
		// returns negative t if it is not valid
	}

	vec3 trace(const Ray& ray, float depth)
	{
		if (depth == 0) return vec3(0, 0, 0);
		Hit hit = firstIntersect(ray);

		if (hit.t < 0)
			return vec3(0.71, 0.75, 1);

		vec3 shadowRayDir, shadowRayOrigin, refRayDir, refRayOrigin;
		vec3 color = vec3(0.05, 0.1, 0.15);
		for (int i = 0; i < lightsources.size(); i++)
		{
			shadowRayDir = lightsources[i]->getLightDirAt(hit.position);
			shadowRayOrigin = hit.position + shadowRayDir * 0.001;
			Hit s = firstIntersect(Ray(shadowRayOrigin, shadowRayDir));

			if (hit.material->isReflective())
			{
				refRayDir = hit.normal * (ray.dir.dot(hit.normal)) * 2 - ray.dir;
				refRayOrigin = hit.position + hit.normal * 0.001;
				color += trace(Ray(refRayOrigin, -refRayDir), depth - 1);
				return color * 0.5 + hit.material->shade(hit.position, hit.normal, lightsources[i]);
			}
			if (s.t < 0 || lightsources[i]->getDistanceFrom(hit.position) < s.t)
			{
				color += hit.material->shade(hit.position, hit.normal, lightsources[i]);
			}
		}
		return color;
	}
};


Scene scene;


bool computeImage()
{
	static unsigned int iPart = 0;

	if (iPart >= 64)
		return false;
	for (int j = iPart; j < windowHeight; j += 64)
	{
		for (int i = 0; i < windowWidth; i++)
		{
			float ndcX = (2.0 * i - windowWidth) / windowWidth;
			float ndcY = (2.0 * j - windowHeight) / windowHeight;
			Camera& camera = scene.getCamera();
			Ray ray = Ray(camera.getEye(), camera.rayDirFromNdc(ndcX, ndcY));

			image[j*windowWidth + i] = scene.trace(ray, 10);
		}
	}
	iPart++;
	return true;
}

void onDisplay() {
	glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (computeImage())
		glutPostRedisplay();
	glDrawPixels(windowWidth, windowHeight, GL_RGB, GL_FLOAT, image);

	glutSwapBuffers();
}



int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition(100, 100);
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow("Ray Casting");

#if !defined(__APPLE__)
	glewExperimental = true;
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	glViewport(0, 0, windowWidth, windowHeight);

	glutDisplayFunc(onDisplay);

	glutMainLoop();

	return 1;
}


/*
whenever we find intersection pt in object, test whether intersection is visible from light source
if visible, cast shadow rays towards pos of light source
if no object in between, take light source into account, otherwise don't

scene has list of light sources
in trace funciton iterate through them
shade calculates elementary contributeions

call algo recursively for reflective rays



assign arbitrary  color to light source (Le)
directional: position is direction
ptlight: subtract position of object from light source position, normalized
(p-l).norm();

abstract light source (w vector) -> pt light (w position) and dir light (w direction)
put light sources into vector of light sources

iterate thru lsources
call shade function for material in hit record
parameterize shade function according to given light source
check if object is visible from lightsource
-> check directionn of (l-p)/magnitude(l-p)
-> add small value to origin towards direction of shadow ray to prevent errors o = o + direction-of-shadow-ray * small-epsilon-value
-> once you have an intersect p, find shadow ray and call intersect to find point s. If distance from p to s is smaller than distance from p to l, light not visible
hit S = firstintersect(shadow-ray); S.t is distance from p to S
shadow-ray = ray(shadowRayOrigin, shadowRayDir)
shadowRayDir = (l-p).normalize();
shadowRayOrigin = p + shadowRayDir * epsilon



get gradient of noise function (see slides)
take original normal vector, add gradient of noise function which depends on position
renormalize normal vector

*/