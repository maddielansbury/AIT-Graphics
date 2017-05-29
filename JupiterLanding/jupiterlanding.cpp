#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>

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

class Platform;
//class Lander;

unsigned int windowWidth = 1280, windowHeight = 720;

bool keyPressed[256];
bool mouseLeftPressed;
enum OBJECT_TYPE { LANDER, PLATFORM, POKEBALL, FLAMETHROWER, DIAMOND, COLL_DIAMOND, LIFE, SKUNTANK, BLACKHOLE};

int majorVersion = 3, minorVersion = 0;

extern "C" unsigned char* stbi_load(
	char const *filename,
	int *x, int *y,
	int *comp, int req_comp);

void getErrorInfo(unsigned int handle)
{
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0)
	{
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

void checkShader(unsigned int shader, char * message)
{
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK)
	{
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

void checkLinking(unsigned int program)
{
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK)
	{
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader 0 in GLSL
const char *vertexSource0 = R"( 
	#version 130 
	precision highp float; 

	in vec2 vertexPosition;		// variable input from Attrib Array 0 selected by glBindAttribLocation
	in vec2 vertexTexCoord;		// variable input from Attrib Array 1 selected by glBindAttribLocation
	out vec2 texCoord;				// output attribute
	uniform mat4 MVP;		// uniform offset
	
	void main() 
	{ 
		texCoord = vertexTexCoord;
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP;  // transform input position by MVP
	} 
)";

// fragment shader 0 in GLSL
const char *fragmentSource0 = R"( 
	#version 130 
	precision highp float; 

	uniform sampler2D samplerUnit;
	in vec2 texCoord; // variable input: interpolated texture coords

	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation 
	
	void main() 
	{
		fragmentColor = texture(samplerUnit, texCoord); // output color is taken from texture
	} 
)";

// vertex shader 1 in GLSL
const char *vertexSource1 = R"( 
	#version 130 
	precision highp float; 

	in vec2 vertexPosition;		// variable input from Attrib Array 0 selected by glBindAttribLocation
	in vec2 vertexTexCoord;		// variable input from Attrib Array 1 selected by glBindAttribLocation
	out vec2 texCoord;				// output attribute
	uniform mat4 MVP;		// uniform offset
	
	void main() 
	{ 
		texCoord = vertexTexCoord;
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP;  // transform input position by MVP
	} 
)";

// fragment shader 1 in GLSL
const char *fragmentSource1 = R"( 
	#version 130 
	precision highp float; 

	uniform sampler2D samplerUnit;
	in vec2 texCoord; // variable input: interpolated texture coords

	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation 
	
	void main() 
	{
		fragmentColor = texture(samplerUnit, texCoord); // output color is taken from texture
	} 
)";

// row-major matrix 4x4
struct mat4
{
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33)
	{
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right)
	{
		mat4 result;
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4
{
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1)
	{
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat)
	{
		vec4 result;
		for (int j = 0; j < 4; j++)
		{
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	vec4 operator+(const vec4& vec)
	{
		vec4 result(v[0] + vec.v[0], v[1] + vec.v[1], v[2] + vec.v[2], v[3] + vec.v[3]);
		return result;
	}
};

// 2D point in Cartesian coordinates
struct vec2
{
	float x, y;

	vec2(float x = 0.0, float y = 0.0) : x(x), y(y) {}

	vec2 operator+(const vec2& v)
	{
		return vec2(x + v.x, y + v.y);
	}

	vec2 operator+=(const vec2& v)
	{
		return vec2(x += v.x, y += v.y);
	}

	vec2 operator-(const vec2& v)
	{
		return vec2(x - v.x, y - v.y);
	}

	vec2 operator-=(const vec2& v)
	{
		return vec2(x -= v.x, y -= v.y);
	}

	vec2 operator*(const vec2& v)
	{
		return vec2(x * v.x, y * v.y);
	}

	vec2 operator*(float f)
	{
		return vec2(x * f, y * f);
	}

	vec2 operator*=(const vec2& v)
	{
		return vec2(x *= v.x, y *= v.y);
	}

	vec2 operator*=(float f)
	{
		return vec2(x *= f, y *= f);
	}

	vec2 unitVector()
	{
		float magnitude;
		magnitude = sqrt(x*x + y*y);
		vec2 unitVec = vec2(x / magnitude, y / magnitude);
		return unitVec;
	}

	static vec2 random() {
		return vec2((((float)rand() / RAND_MAX) - 0.5)*2.0, (((float)rand() / RAND_MAX) - 0.5)*2.0);
	}
};


// handle of the shader program
unsigned int shaderProgram0;
unsigned int shaderProgram1;
boolean skuntankCaught;
vec2 landerPos;
vec2 landerVel;
vec2 clickPos;
vec2 gravity = vec2(0.0, -0.5);

class Object {
protected:
	vec2 scale; float orientation; vec2 position; vec2 velocity; vec2 acceleration; float angularVelocity; float angularAcceleration;
	vec2 vertCoords[2]; Object *parent;
	boolean destroyed;
	unsigned int vao;
	unsigned int shader;
public:
	Object(unsigned int sp) : scale(1.0, 1.0), orientation(0.0), position(0.0, 0.0),
		velocity(0.0, 0.0), acceleration(0.0, 0.0), angularVelocity(0.0), angularAcceleration(0.0),
		destroyed(false), shader(sp) {
		vertCoords[0] = position;
		vertCoords[1] = vec2(position.x + scale.x, position.y + scale.y);
	}

	Object(unsigned int sp, Object *parent) : scale(1.0, 1.0), orientation(0.0), position(0.0, 0.0),
		velocity(0.0, 0.0), acceleration(0.0, 0.0), angularVelocity(0.0), angularAcceleration(0.0),
		destroyed(false), shader(sp), parent(parent) {
		vertCoords[0] = position;
		vertCoords[1] = vec2(position.x + scale.x, position.y + scale.y);
	}

	Object* Scale(const vec2& s) { scale *= s; return this; }
	Object* Rotate(float angle) { orientation += angle; return this; }
	Object* Translate(const vec2& t) { position += t; return this; }

	void SetTransform() {
		mat4 MVPTransform;
		mat4 MTranslation(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			position.x, position.y, 0, 1
		);
		mat4 MScale(
			scale.x, 0, 0, 0,
			0, scale.y, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		);
		mat4 MRotate(
			cos(orientation), sin(orientation), 0, 0,
			-sin(orientation), cos(orientation), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		);
		
		mat4 VTranslation(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-landerPos.x, -landerPos.y, 0, 1
		);

		mat4 VScale(
			1.0, 0, 0, 0,
			0, (float)windowWidth / windowHeight, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		MVPTransform = MScale * MRotate * MTranslation * VTranslation * VScale; 

	/*	mat4 V(
			1.0, 0, 0, 0,
			0, (float)windowWidth / windowHeight, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
		MVPTransform = MScale * MRotate * MTranslation * V;*/

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shader, "MVP");
		// set uniform variable MVP to the MVPTransform
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform);
		else printf("uniform MVPTransform cannot be set\n");
	}

	void Draw() {
		glUseProgram(shader);
		SetTransform();
		DrawModel();
	}

	vec2 getPosition() {
		return position;
	}

	vec2 getVelocity() {
		return velocity;
	}

	boolean isDestroyed() {
		return destroyed;
	}

	virtual void Move(float) = 0;
	virtual void Control() = 0;
	virtual void DrawModel() = 0;
	virtual void Interact(Object* o) = 0;
	virtual OBJECT_TYPE GetType() = 0;
};

class Texture {
	unsigned int textureId;
public:
	Texture(const std::string& inputFileName) {
		unsigned char* data; int width; int height; int nComponents = 4;
		// load texture from file
		data = stbi_load(inputFileName.c_str(), &width, &height, &nComponents, 0);
		if (data == NULL) { return; }

		glGenTextures(1, &textureId); // generate texture ID
		glBindTexture(GL_TEXTURE_2D, textureId); // make the given texture active

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data); // upload the texture to the GPU memory

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // set minification filter
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // set magnification filter
		delete data; // delete the temporary buffer
	}

	void Bind(unsigned int shader) {
		int samplerUnit = 0;
		int location = glGetUniformLocation(shader, "samplerUnit");
		glUniform1i(location, samplerUnit);
		glActiveTexture(GL_TEXTURE0 + samplerUnit);
		glBindTexture(GL_TEXTURE_2D, textureId);
	}
};


class TexturedQuad : public Object {
	Texture *texture;

public:
	TexturedQuad(Texture* t, unsigned int sp = shaderProgram0) : Object(sp), texture(t)
	{
		glGenVertexArrays(1, &vao);	// create a vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);		// generate 2 vertex buffer objects

										// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5 };	// vertex data on the CPU

		glBufferData(GL_ARRAY_BUFFER,	// copy to the GPU
			sizeof(vertexCoords),	// size of the vbo in bytes
			vertexCoords,		// address of the data array on the CPU
			GL_STATIC_DRAW);	// copy to that part of the memory which is not modified 

								// map Attribute Array 0 to the currently bound vertex buffer (vbo)
		glEnableVertexAttribArray(0);

		// data organization of Attribute Array 0 
		glVertexAttribPointer(0,	// Attribute Array 0 
			2, GL_FLOAT,	// components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);		// stride and offset: it is tightly packed

							// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexTexCoord[] = { 0, 0, 1, 0, 1, 1, 0, 1 }; // vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexTexCoord), vertexTexCoord, GL_STATIC_DRAW); // copy to the GPU
																							   // map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // data organization of Attribute Array 1
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	}

	void DrawModel()
	{
		texture->Bind(shader);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBindVertexArray(vao);
		glDrawArrays(GL_QUADS, 0, 4);
		glDisable(GL_BLEND);
	}
};

class TransparentTexturedQuad : public Object {
	Texture *texture;

public:
	TransparentTexturedQuad(Texture* t, unsigned int sp = shaderProgram1) : Object(sp), texture(t)
	{
		glGenVertexArrays(1, &vao);	// create a vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);		// generate 2 vertex buffer objects

										// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5 };	// vertex data on the CPU

		glBufferData(GL_ARRAY_BUFFER,	// copy to the GPU
			sizeof(vertexCoords),	// size of the vbo in bytes
			vertexCoords,		// address of the data array on the CPU
			GL_STATIC_DRAW);	// copy to that part of the memory which is not modified 

								// map Attribute Array 0 to the currently bound vertex buffer (vbo)
		glEnableVertexAttribArray(0);

		// data organization of Attribute Array 0 
		glVertexAttribPointer(0,	// Attribute Array 0 
			2, GL_FLOAT,	// components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);		// stride and offset: it is tightly packed

							// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexTexCoord[] = { 0, 0, 1, 0, 1, 1, 0, 1 }; // vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexTexCoord), vertexTexCoord, GL_STATIC_DRAW); // copy to the GPU
																							   // map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // data organization of Attribute Array 1
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	}

	void DrawModel()
	{
		texture->Bind(shader);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBindVertexArray(vao);
		glDrawArrays(GL_QUADS, 0, 4);
		glDisable(GL_BLEND);
	}
};

class Platform : public TexturedQuad
{
public:
	Platform(Texture* t) : TexturedQuad(t)
	{
		scale = vec2(0.1, 0.2);
		orientation = M_PI/2;
		position = vec2::random();
		velocity = vec2(0.0, 0.0);
		angularVelocity = 0.0;
	}

	void Interact(Object* o) {}

	OBJECT_TYPE GetType()
	{
		return PLATFORM;
	}

	void Move(float dt) {}
	void Control() {}
};

class BlackHole : public TexturedQuad
{
public:
	BlackHole(Texture* t) : TexturedQuad(t)
	{
		scale = vec2(0.2, 0.2);
		position = vec2::random();
		velocity = vec2(0.0, 0.0);
		angularVelocity = 0.0;
	}

	void Interact(Object* o) {}

	OBJECT_TYPE GetType()
	{
		return BLACKHOLE;
	}

	void Move(float dt) {}
	void Control() {}
};

class Pokeball : public TexturedQuad
{
public:
	Pokeball(Texture* t) : TexturedQuad(t)
	{
		scale = vec2(0.05, 0.05);
		position = landerPos;
		orientation = M_PI;
		angularVelocity = 0.0;
		acceleration = gravity;
		velocity = (position - clickPos).unitVector();
		destroyed = false;
	}

	void Move(float dt)
	{
		velocity += acceleration * dt;
		
		position += velocity * dt;
		if (abs(position.x) > abs(landerPos.x) + 1 ||
			abs(position.y) > abs(landerPos.y) + 1 * (float)windowWidth / windowHeight)
			destroyed = true;
	}

	void Interact(Object* o) {}

	OBJECT_TYPE GetType()
	{
		return POKEBALL;
	}

	void Control() {}	
};

class Flamethrower : public TransparentTexturedQuad
{
	int lifetime;
public:
	Flamethrower(Texture* t) : TransparentTexturedQuad(t)
	{
		lifetime = 0;
		scale = vec2(0.05, 0.05);
		position = landerPos;
		orientation = M_PI;
		angularVelocity = 0.0;
		acceleration = 0.0;
		velocity = (position - clickPos).unitVector();
		destroyed = false;
	}

	void Move(float dt)
	{
		if (lifetime >= 150) destroyed = true;
		velocity += acceleration * dt;

		position += velocity * dt;
		if (abs(position.x) > abs(landerPos.x) + 1 ||
			abs(position.y) > abs(landerPos.y) + 1 * (float)windowWidth / windowHeight)
			destroyed = true;

		lifetime++;
	}

	void Interact(Object* o) {}

	OBJECT_TYPE GetType()
	{
		return FLAMETHROWER;
	}

	void Control() {}
};

int diamonds = 0;
int diamondIndex = 0;
int lives = 5;
int lifeIndex = 0;

class Diamond : public TexturedQuad
{
public:
	Diamond(Texture* t) : TexturedQuad(t)
	{
		scale = vec2(0.1, 0.1);
		position.x = (landerPos.x - 1) + ((float)rand() / RAND_MAX) * 2.0;
		position.y = abs(landerPos.y) + 1 * (float)windowWidth / windowHeight;
		orientation = M_PI;
		angularVelocity = 0.0;
		acceleration = gravity;
		velocity = vec2(0.0, 0.0);
		destroyed = false;
	}

	void Move(float dt)
	{
		velocity *= exp(-dt);
		velocity += acceleration * dt;
		position += velocity * dt;
		if (abs(position.x) > abs(landerPos.x) + 1 ||
			abs(position.y) > abs(landerPos.y) + 1 * (float)windowWidth / windowHeight)
			destroyed = true;
	}

	void Collect()
	{
		destroyed = true;
		diamonds++;
		diamondIndex++;
	}


	void Interact(Object* o) {}

	OBJECT_TYPE GetType()
	{
		return DIAMOND;
	}

	void Control() {}
};

class CollectedDiamond : public TexturedQuad
{
	int num;
public:
	CollectedDiamond(Texture* t, int n) : TexturedQuad(t)
	{
		scale = vec2(0.05, 0.05);
		orientation = M_PI;
		angularVelocity = 0.0;
		velocity = landerVel;
		destroyed = false;
		num = n;
	}

	void Move(float dt) {
		position.x = landerPos.x + 1.0 - 0.05 * num;
		position.y = landerPos.y + 1.0 / ((float)windowWidth / windowHeight) - 0.05;
	}

	void Interact(Object* o) {}

	OBJECT_TYPE GetType()
	{
		return COLL_DIAMOND;
	}

	void Control() {}
};

class Life : public TexturedQuad
{
	int num;
public:
	Life(Texture* t, int n) : TexturedQuad(t)
	{
		scale = vec2(0.05, 0.05);
		orientation = M_PI;
		angularVelocity = 0.0;
		velocity = landerVel;
		destroyed = false;
		num = n;
	}

	void Move(float dt) {
		position.x = landerPos.x - 1.0 + 0.05 * num;
		position.y = landerPos.y - 1.0 / ((float)windowWidth / windowHeight) + 0.05;
	}

	void Interact(Object* o) {}

	OBJECT_TYPE GetType()
	{
		return LIFE;
	}

	void Control() {}
};

class Skuntank : public TexturedQuad
{
public:
	Skuntank(Texture* t) : TexturedQuad(t)
	{
		scale = vec2(0.3, 0.2);
		orientation = M_PI;
		position = vec2::random();
		velocity = vec2(0.0, 0.0);
		angularVelocity = 0.0;
	}

	void Interact(Object* o)
	{
		vec2 difference;
		difference = position - o->getPosition();
		if (o->GetType() == POKEBALL)
		{
			if ((difference.x < 0.25 && difference.x > -0.25) &&
				(difference.y < 0.18 && difference.y > -0.18))
			{
				skuntankCaught = true;
				destroyed = true;
			}
		}
	}

	OBJECT_TYPE GetType()
	{
		return SKUNTANK;
	}

	void Move(float dt) {}
	void Control() {}
};

class Lander : public TexturedQuad
{

public:
	Lander(Texture* t) : TexturedQuad(t)
	{
		scale = vec2(0.2, 0.2);
		position = vec2(0.0, 0.0);
		acceleration = gravity;
		angularAcceleration = 0.0;
		angularVelocity = 0.0;
		orientation = M_PI;
		destroyed = false;
	}

	void Control()
	{
		if (keyPressed['a'])
		{
			velocity += vec2(-0.001, 0.0);
			angularVelocity += 0.001;
		}
		if (keyPressed['d'])
		{
			velocity += vec2(0.001, 0.0);
			angularVelocity -= 0.001;
		}
		if (keyPressed['w']) velocity += vec2(0.0, 0.001);
	}

	void Move(float dt)
	{
		velocity *= exp(-dt);
		velocity += acceleration * dt;
		position += velocity * dt;
		angularVelocity *= exp(-dt * 7.0);
		angularVelocity += angularAcceleration * dt;
		orientation += angularVelocity * dt;
	}

	void Interact(Object* o)
	{
		vec2 difference;
		difference = position - o->getPosition();
		if (o->GetType() == PLATFORM)
		{
			if ((difference.x < 0.18 && difference.x > -0.18) &&
				(difference.y < 0.145 && difference.y > -0.145))
			{
				vec2 collisionVec = o->getPosition() - position;
				velocity -= collisionVec * 0.05;
			}
		}

		if (o->GetType() == DIAMOND)
		{
			if ((difference.x < 0.145 && difference.x > -0.145) &&
				(difference.y < 0.145 && difference.y > -0.145))
			{
				((Diamond*)o)->Collect();
			}
		}

		if (o->GetType() == BLACKHOLE)
		{
			if ((difference.x < 0.3 && difference.x > -0.3) &&
				(difference.y < 0.3 && difference.y > -0.3))
			{
				vec2 collisionVec = o->getPosition() - position;
				velocity += collisionVec * 0.005;
			}
		}
	}

	OBJECT_TYPE GetType()
	{
		return LANDER;
	}
};


class Scene
{
	std::vector<Texture*> textures;
	std::vector<Object*> objects;
	Lander *lander;
	Skuntank *skuntank;
	int pokeballCooldown, diamondCooldown, flamethrowerCooldown = 0;
	const static int NUM_PLATFORMS = 10;
	const static int NUM_BLACKHOLES = 3;
public:
	void Initialize()
	{
		textures.push_back(new Texture("lander.png"));
		textures.push_back(new Texture("platform.png"));
		textures.push_back(new Texture("pokeball.png"));
		textures.push_back(new Texture("diamond.png"));
		textures.push_back(new Texture("fireball.png"));
		textures.push_back(new Texture("skuntank.png"));
		textures.push_back(new Texture("blackhole.png"));
		objects.push_back(lander = new Lander(textures[0]));
		objects.push_back(skuntank = new Skuntank(textures[5]));
		for (int i = 0; i < NUM_PLATFORMS; i++) objects.push_back(new Platform(textures[1]));
		for (int i = 0; i < NUM_BLACKHOLES; i++) objects.push_back(new BlackHole(textures[6]));
	}

	~Scene()
	{
		for (int i = 0; i < textures.size(); i++) delete textures[i];
		for (int i = 0; i < objects.size(); i++) delete objects[i];
	}

	void Draw()
	{
		for (int i = 0; i < objects.size(); i++) objects[i]->Draw();
		landerPos = lander->getPosition();
		landerVel = lander->getVelocity();
		pokeballCooldown--;
		flamethrowerCooldown--;
		diamondCooldown--;
	}

	void Move(float dt)
	{
		for (int i = 0; i < objects.size(); i++) objects[i]->Move(dt);
	}

	void Control()
	{
		for (int i = 0; i < objects.size(); i++) objects[i]->Control();
		if (mouseLeftPressed)
		{
			if (!skuntankCaught && pokeballCooldown <= 0)
			{
				objects.push_back(new Pokeball(textures[2]));
				pokeballCooldown = 750;
			} else if (skuntankCaught && flamethrowerCooldown <= 0) {
				objects.push_back(new Flamethrower(textures[4]));
				flamethrowerCooldown = 10;
			}
		}
	}

	void HandleCollisions()
	{
		for (int i = 1; i < objects.size(); i++) {
			lander->Interact(objects[i]);
		}
		for (int i = 1; i < objects.size(); i++) {
			skuntank->Interact(objects[i]);
		}
	}

	void Destroy()
	{
		std::vector<Object*> objectsTemp;
		for (int i = 0; i < objects.size(); i++)
			if (!(objects[i]->isDestroyed())) objectsTemp.push_back(objects[i]);
		objects = objectsTemp;
	}

	void rainDiamonds()
	{
		if (diamondCooldown <= 0)
		{
			objects.push_back(new Diamond(textures[3]));
			diamondCooldown = 400;
		}
	}

	void collectDiamonds()
	{
		if (diamonds > 0)
		{
			objects.push_back(new CollectedDiamond(textures[3], diamondIndex));
			diamonds--;
		}
	}
};

Object* object;
Scene scene;

// initialization, create an OpenGL context
void onInitialization()
{
	glViewport(0, 0, windowWidth, windowHeight);

	for (int i = 0; i < 256; i++) {
		keyPressed[i] = false;
	}
		mouseLeftPressed = false;

	// VERTEX SHADER 0
	// create vertex shader 0 from string
	unsigned int vertexShader0 = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader0) { printf("Error in vertex shader 0 creation\n"); exit(1); }

	glShaderSource(vertexShader0, 1, &vertexSource0, NULL);
	glCompileShader(vertexShader0);
	checkShader(vertexShader0, "Vertex shader 0 error");

	// create fragment shader 0 from string
	unsigned int fragmentShader0 = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader0) { printf("Error in fragment shader 0 creation\n"); exit(1); }

	glShaderSource(fragmentShader0, 1, &fragmentSource0, NULL);
	glCompileShader(fragmentShader0);
	checkShader(fragmentShader0, "Fragment shader 0 error");

	// attach shaders to a single program
	shaderProgram0 = glCreateProgram();
	if (!shaderProgram0) { printf("Error in shader program 0 creation\n"); exit(1); }

	glAttachShader(shaderProgram0, vertexShader0);
	glAttachShader(shaderProgram0, fragmentShader0);

	// connect Attrib Array to input variables of the vertex shader
	glBindAttribLocation(shaderProgram0, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
	glBindAttribLocation(shaderProgram0, 1, "vertexColor"); // vertexColor gets values from Attrib Array 1


															// connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram0, 0, "fragmentColor"); // fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram0);
	checkLinking(shaderProgram0);
	// make this program run
	glUseProgram(shaderProgram0);

	// VERTEX SHADER 1
	// create vertex shader 1 from string
	unsigned int vertexShader1 = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader1) { printf("Error in vertex shader 0 creation\n"); exit(1); }

	glShaderSource(vertexShader1, 1, &vertexSource1, NULL);
	glCompileShader(vertexShader1);
	checkShader(vertexShader1, "Vertex shader 0 error");

	// create fragment shader 0 from string
	unsigned int fragmentShader1 = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader1) { printf("Error in fragment shader 0 creation\n"); exit(1); }

	glShaderSource(fragmentShader1, 1, &fragmentSource1, NULL);
	glCompileShader(fragmentShader1);
	checkShader(fragmentShader1, "Fragment shader 0 error");

	// attach shaders to a single program
	shaderProgram1 = glCreateProgram();
	if (!shaderProgram1) { printf("Error in shader program 0 creation\n"); exit(1); }

	glAttachShader(shaderProgram1, vertexShader1);
	glAttachShader(shaderProgram1, fragmentShader1);

	// connect Attrib Array to input variables of the vertex shader
	glBindAttribLocation(shaderProgram1, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
	glBindAttribLocation(shaderProgram1, 1, "vertexColor"); // vertexColor gets values from Attrib Array 1


															// connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram1, 0, "fragmentColor"); // fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram1);
	checkLinking(shaderProgram1);
	// make this program run
	glUseProgram(shaderProgram1);

	scene.Initialize();

	/* UNIFORM COLOR
	vec4 vertexColor(1, 0, 0, 1); // constant red color
	int location = glGetUniformLocation(shaderProgram, "vertexColor");
	if (location >= 0) glUniform3fv(location, 1, &vertexColor.v[0]); // set uniform variable vertexColor
	else printf("uniform vertex color cannot be set\n");
	*/
}

void onExit()
{
	delete object;
	glDeleteProgram(shaderProgram0);
	glDeleteProgram(shaderProgram1);
	printf("exit");
}

// window has become invalid: redraw
void onDisplay()
{
	glClearColor(0, 0, 0, 0); // background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	scene.Draw();
	glViewport(0, 4 / 5 * windowHeight, windowWidth / 5, windowHeight / 5);
	scene.Draw();
	glViewport(0, 0, windowWidth, windowHeight);

	glutSwapBuffers(); // exchange the two buffers
}

void onReshape(int winWidth0, int winHeight0)
{
	windowHeight = winHeight0;
	windowWidth = winWidth0;
	glViewport(0, 0, winWidth0, winHeight0);
}

void onKeyboard(unsigned char key, int xmouse, int ymouse)
{
	keyPressed[key] = true;
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int xmouse, int ymouse)
{
	keyPressed[key] = false;
	glutPostRedisplay();
}

void onMouse(int button, int state, int x, int y)
{
	float xf = x;
	float yf = y;
	mouseLeftPressed = (state == GLUT_DOWN);
	clickPos = vec2((float)windowWidth/2.0 - xf - 1.0, -((float)windowHeight/2.0 - yf - 1.0));
	glutPostRedisplay();
}


void onIdle() {

	// time elapsed since program started, in seconds
	double t = glutGet(GLUT_ELAPSED_TIME) * 0.001;
	// variable to remember last time idle was called
	static double lastTime = 0.0;
	// time difference between calls: time step  
	double dt = t - lastTime;
	// store time
	lastTime = t;

	scene.rainDiamonds();
	scene.collectDiamonds();
	scene.Destroy();
	scene.HandleCollisions();
	scene.Move(dt);
	scene.Control();

	glutPostRedisplay();
}


int main(int argc, char * argv[])
{
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight); 	// application window is initially of resolution 512x512
	glutInitWindowPosition(50, 50);			// relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow("Triangle Rendering");

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

	onInitialization();

	glutDisplayFunc(onDisplay); // register event handlers
	glutReshapeFunc(onReshape);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);

	glutMainLoop();
	onExit();
	return 1;
}