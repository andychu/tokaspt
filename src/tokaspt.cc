/*
	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License version 2
	as published by the Free Software Foundation.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA


	Copyright (C) 2009  Thierry Berger-Perrin <tbptbp@gmail.com>, http://ompf.org
*/
// because of some quirks, it's better if we know.
// #define USE_FREEGLUT

#include "specifics.h"
#include "tweaks.h"

#include "math_linear.h"
#include "rt_sphere.h"

#include "gl_scene.h"

#include "smallpt.h"


#include <cstdarg>
#include <cstring>		// mem ops.
#include <cassert>
#include <cstdio>
#include <cstdlib>		// realloc, free
#include <new> 		// placement new.
#include <vector>

#include <algorithm> // std::min/max

#include "gl_camera.h"

#include <GL/glew.h>
#ifndef USE_FREEGLUT
	#include <GL/glut.h>
#else
	#include <GL/freeglut.h>
#endif

#ifdef _MSC_VER
	#pragma comment(lib, "glut32.lib")
	#pragma comment(lib, "glew32.lib")
#endif

#include <cuda_gl_interop.h>
#include <cutil_gl_error.h>

#if 0
	// cruft removal, there's a disable but of course no enable.
	#pragma warning(push, 4)
	#pragma warning(disable : 4201 4127 4100 4245) // non standard, conditional expression is constant, unreferenced formal parameter, signed / unsigned
#endif


// initial window size
enum { window_width = 3*tile_size_x, window_height = 2*tile_size_y };

static const char glut_init_display_string[] =
	#ifndef FREEGLUT
		// "rgba double depth>=16 samples>=1"; // no limit, gets 16 samples, but it's ssslllooowww here.
		"rgba double depth>=16 samples"; // 4 samples.
		// "rgba double depth>=16 samples=0";
	#else
		// FreeGlut doesn't like depth>=16
		//FIXME: doesn't give use any multisampling either.
		"rgba double depth samples";
	#endif

// ===========================================================================
//
//							various utilities.
//
// ===========================================================================

// get rid of stupid warnings.
template<typename T, typename U, typename V>
	static T clamp(U low, V high, T val) { return val < T(low) ? low : val > T(high) ? high : val; }

namespace misc {
	// glorious naming.
	template<typename T> static T argument_reduction(const T limit, const T step, T val) {  while (val > limit) val -= step; return val; }

	template<typename T> static void wipe(T &m) { std::memset(&m, 0, sizeof(T)); }

	static const char *refl2string(refl_t refl) {
		const char *txt[4] = { "DIFF", "SPEC", "REFR", "BOGUS" };
		unsigned i = unsigned(refl);
		return txt[i < 4 ? i : 3];
	}
}

namespace sys {
	fmt_t::fmt_t(const char * __restrict const fmt, ...) {
		va_list vl; va_start(vl, fmt);
			len = vsnprintf(buffer, buffer_size, fmt, vl);
			buffer[buffer_size-1] = 0;
		va_end(vl);
	}
}

void fatal(const char * const msg) {
	std::fprintf(stderr, "*** fatal: %s\n", msg);
	std::abort();
}


// ===========================================================================
//						CUDA interop
// ===========================================================================
static scene_t scene;

static void *renderer = 0;
static render_params_t params;

static void smallpt_reset_accumulator() {
	params.pass = 0;
}

// ===========================================================================
//						glut callbacks
// ===========================================================================
void display();
void idle();
template<bool down> void keyboard_regular(unsigned char key, int x, int y);
template<bool down> void keyboard_special(int key, int x, int y);
void reshape(int w, int h);
void menu(int i);
void mouse(int x, int y, int z, int w);
void motion(int x, int y);





namespace gl { class camera_t; }
namespace ui {
	static void init();
	static void update(const float time_abs, const float time_delta, gl::camera_t &camera);
	static void display(const float time_abs, const float time_delta);
}


namespace gl {
	const vec_t camera_t::world_ups[6] = {
		vec_t(0,+1,0), vec_t(0,0,+1), vec_t(+1,0,0),
		vec_t(0,-1,0), vec_t(0,0,-1), vec_t(-1,0,0)
	};

	struct texture_and_pbo_t;

	// ===========================================================================
	//					various states
	// ===========================================================================
	static GLint current_width, current_height;

	static unsigned timer;				// a wall clock.
	static float last_render_time;		// last cuda render time.
	static unsigned show_ui = false;	// full or minimal 3D bloat.

	namespace state {
		static camera_t cam, prev_cam;			// so we can detect dirtyness.
		static screen_sampler_t smp, prev_smp;	// ditto

		static void copy_camera() {
			std::memcpy(&prev_cam, &cam, sizeof(cam));
			std::memcpy(&prev_smp, &smp, sizeof(smp));
		}
		static bool_t is_camera_dirty() {
			return
				std::memcmp(&cam, &prev_cam, sizeof(cam)) != 0 ||
				std::memcmp(&smp, &prev_smp, sizeof(smp)) != 0;
		}

		static texture_and_pbo_t *tex_and_pbo = 0;
	}
}

namespace gl {
	namespace tex {
		template<GLenum format_int, GLenum format, GLenum type>
			static void create(GLuint *id, const point_t res) {
				glGenTextures(1, id);
				glBindTexture(GL_TEXTURE_2D, *id);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				glTexImage2D(GL_TEXTURE_2D, 0, format_int, res.x, res.y, 0, format, type, NULL);
				CUT_CHECK_ERROR_GL();
			}
		static void destroy(GLuint *id) {
			glDeleteTextures(1, id);
			CUT_CHECK_ERROR_GL();
			*id = ~0u;
		}
	}
	namespace pbo {
		template<GLenum usage>
			static void create(GLuint *id, size_t data_size) {
				glGenBuffers(1, id);
				GLenum target = GL_ARRAY_BUFFER /* GL_PIXEL_UNPACK_BUFFER_ARB */;
				glBindBuffer(target, *id);
				glBufferData(target, data_size, 0, usage);
				cutilSafeCall(cudaGLRegisterBufferObject(*id)); // register once
				CUT_CHECK_ERROR_GL();
			}
		static void destroy(GLuint *id) {
			cutilSafeCall(cudaGLUnregisterBufferObject(*id));
			glBindBuffer(GL_ARRAY_BUFFER, *id);
			glDeleteBuffers(1, id);
			CUT_CHECK_ERROR_GL();
			*id = ~0u;
		}
	}

	// holds a pbo, a texture and ensure correct cuda registration/mapping.
	// centralized because we need to maintain them in sync and dynamically resize.
	struct texture_and_pbo_t {
		GLuint tex_id, pbo_id;
		size_t data_size;
		point_t res;
		typedef PIXEL_CPNT_TYPE component_t;
		enum {
			num_components = 3, component_size = sizeof(component_t),
			/* format_int = GL_RGBA, format = GL_RGBA, format_pbo = GL_BGRA, */
			format_int = GL_RGB8,
			format     = GL_RGB,
			format_pbo = GL_RGB,
			type = PIXEL_GL_TYPE
		};
		texture_and_pbo_t(const point_t &resolution)
			:
				tex_id(-1), pbo_id(-1), data_size(resolution.area()*num_components*component_size),
				res(resolution)
			{
				gl::tex::create<format_int, format, type>(&tex_id, res);
				gl::pbo::create<GL_STREAM_COPY /* GL_DYNAMIC_DRAW */>(&pbo_id, data_size);

			}
		~texture_and_pbo_t() {
			glBindTexture(GL_TEXTURE_2D, 0);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			gl::pbo::destroy(&pbo_id);
			gl::tex::destroy(&tex_id);
		}

		// give cuda a pbo to fill, upload a texture with it.
		float render(render_params_t * const params) const {
			cudaGLMapBufferObject((void**) &params->framebuffer, pbo_id);
				float dt = smallpt_render(renderer, params);
			cudaGLUnmapBufferObject(pbo_id);
			// now upload that back to the texture.
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_id);
			glBindTexture(GL_TEXTURE_2D, tex_id);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, res.x, res.y, format_pbo, type, NULL);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			CUT_CHECK_ERROR_GL();
			return dt;
		}

		// reallocate everything (cuda stuff, texture, pbo) if needed.
		static bool_t resize_everything_if(const point_t &res, void *&rdr, texture_and_pbo_t *&tab) {
			if (!rdr || !tab || tab->res != res) {
				printf("resize_everything_if: allocation for (%d,%d)\n",res.x, res.y);
				if (rdr) smallpt_destroy(rdr);
				rdr = smallpt_make(res.x, res.y);
				delete tab;
				tab = new texture_and_pbo_t(res);
				return true;
			}
			else return false;
		}
	};




	namespace misc {
		static void draw_grid(float grid_size, float step) {
			static GLuint l_grid = -1;
			// parameters don't change anyway.
			if (l_grid == -1u) {
				l_grid = glGenLists(1);
				glNewList(l_grid, GL_COMPILE);
					// best result with if really attenuated.
					glBegin(GL_LINES);
						for(float i=step; i <= grid_size; i+= step) {
							glVertex3f(-grid_size, 0,  i); glVertex3f( grid_size, 0,  i);
							glVertex3f(-grid_size, 0, -i); glVertex3f( grid_size, 0, -i);
							glVertex3f( i, 0, -grid_size); glVertex3f( i, 0,  grid_size);
							glVertex3f(-i, 0, -grid_size); glVertex3f(-i, 0,  grid_size);
						}

						// flash some axis.
						glColor3f(0.5f, 0, 0);
						glVertex3f(-grid_size, 0, 0); glVertex3f(grid_size, 0, 0);
						glColor3f(0,0,0.5f);
						glVertex3f(0, 0, -grid_size); glVertex3f(0, 0,  grid_size);
					glEnd();
				glEndList();
			}
			else
				glCallList(l_grid);

		}

		static void draw_axis(float size) {
			const float line_w = 3.25, point_w = 6.25;
			glPointSize(point_w);
			glBegin(GL_POINTS);
				glColor3f(1, 0, 0);
				glVertex3f(size, 0, 0);
				glColor3f(0, 1, 0);
				glVertex3f(0, size, 0);
				glColor3f(0, 0, 1);
				glVertex3f(0, 0, size);
			glEnd();
			glPointSize(1);
			glLineWidth(line_w);
			glBegin(GL_LINES);
				glColor3f(1, 0, 0);
				glVertex3f(0, 0, 0); glVertex3f(size, 0, 0);
				glColor3f(0, 1, 0);
				glVertex3f(0, 0, 0); glVertex3f(0, size, 0);
				glColor3f(0, 0, 1);
				glVertex3f(0, 0, 0); glVertex3f(0, 0, size);
			glEnd();
			glLineWidth(1);
		}

		// glut stroke stuff
		//
		// GLUT_STROKE_ROMAN
		//    A proportionally spaced Roman Simplex font for ASCII characters 32 through 127.
		//    The maximum top character in the font is 119.05 units; the bottom descends 33.33 units.
		// GLUT_STROKE_MONO_ROMAN
		//    A mono-spaced spaced Roman Simplex font (same characters as GLUT_STROKE_ROMAN) for ASCII characters 32 through 127.
		//    The maximum top character in the font is 119.05 units; the bottom descends 33.33 units. Each character is 104.76 units wide.
		namespace stroke {
			/* doesn't bring anything, i suppose Glut already has lists.
			enum { num_glyphs = 128 };
			static GLuint l_glyph_base = -1;
			*/
			#if 0
				static const float glyph_h = 119.05+33.33, glyph_w = 120;
				// static const float scale = 1.f/glyph_w;
				static void *font = GLUT_STROKE_ROMAN;
			#else
				static const float glyph_h = 119.05f+33.33f, glyph_w = 104.76f;
				// static const float scale = 1.f/glyph_w;
				static void *font = GLUT_STROKE_MONO_ROMAN;
			#endif
			static void init() {
				/*
				l_glyph_base = glGenLists(128);
				for (unsigned i=0; i<num_glyphs; ++i) {
					glNewList(l_glyph_base + i, GL_COMPILE);
						glutStrokeCharacter(font, i);
					glEndList();
				}
				*/
			}

			static void print(const char * const txt) {
				if (!txt) return;
				for (const char *p=txt; *p; ++p) {
					glutStrokeCharacter(font, *p);
					// assert(*p < num_glyphs && "glyph overflow");  glCallList(l_glyph_base + *p);
				}
			}
			static int width(const char * const txt) {
				int w = 0;
				for (const char *p=txt; *p; ++p)
					w += glutStrokeWidth(font, *p);
				return w;
			}
		}

		#if 0
			#define GLUT_BITMAP_9_BY_15		((void*)2)
			#define GLUT_BITMAP_8_BY_13		((void*)3)
			#define GLUT_BITMAP_TIMES_ROMAN_10	((void*)4)
			#define GLUT_BITMAP_TIMES_ROMAN_24	((void*)5)
			#define GLUT_BITMAP_HELVETICA_10	((void*)6)
			#define GLUT_BITMAP_HELVETICA_12	((void*)7)
			#define GLUT_BITMAP_HELVETICA_18	((void*)8)
		#endif

		namespace bitmap {
			// ah crap, there's a baseline.
			// static void *font = GLUT_BITMAP_HELVETICA_12;
			static void *font = GLUT_BITMAP_8_BY_13;
			//static void *font = GLUT_BITMAP_HELVETICA_10;
			static int line_height = 14, descent_nudge = 4;
			static void print(const char c) { glutBitmapCharacter(font, c); }
			static void print(const char * const txt) {
				for (const char *p=txt; *p; ++p)
					glutBitmapCharacter(font, *p);
			}
			static int width(const char * const txt) {
				int w = 0;
				for (const char *p=txt; *p; ++p)
					w += glutBitmapWidth(font, *p);
				return w;
			}
		}
	}


	// ===========================================================================
	//				where we finally draw stuff.
	// ===========================================================================
	namespace geosphere {
		static const float X = .525731112119133606f, Z = .850650808352039932f;
		static const GLfloat vdata[12][3] = {
			{-X, 0, Z}, { X, 0,  Z}, {-X,  0, -Z}, { X,  0, -Z},
			{ 0, Z, X}, { 0, Z, -X}, { 0, -Z,  X}, { 0, -Z, -X},
			{ Z, X, 0}, {-Z, X,  0}, { Z, -X,  0}, {-Z, -X,  0}
		};
		static const GLuint tindices[20][3] = {
			{0, 4, 1}, {0,9, 4}, {9, 5,4}, { 4,5,8}, {4,8, 1},
			{8,10, 1}, {8,3,10}, {5, 3,8}, { 5,2,3}, {2,7, 3},
			{7,10, 3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1, 6},
			{6, 1,10}, {9,0,11}, {9,11,2}, { 9,2,5}, {7,2,11} };

		static void normalize(GLfloat *a) {
			GLfloat d = 1/std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
			a[0] *= d;
			a[1] *= d;
			a[2] *= d;
		}

		static unsigned draw_triangles(const GLfloat a[3], const GLfloat b[3], const GLfloat c[3], const int div, const float r) {
			if (div <= 0) {
				glVertex3f(a[0]*r, a[1]*r, a[2]*r);
				glVertex3f(b[0]*r, b[1]*r, b[2]*r);
				glVertex3f(c[0]*r, c[1]*r, c[2]*r);
				return 1;
			}
			else {
				GLfloat ab[3], ac[3], bc[3];
				for (unsigned i=0; i<3; ++i) {
					ab[i] = (a[i]+b[i])/2;
					ac[i] = (a[i]+c[i])/2;
					bc[i] = (b[i]+c[i])/2;
				}
				normalize(ab); normalize(ac); normalize(bc);
				return
					draw_triangles( a, ab, ac, div-1, r) +
					draw_triangles( b, bc, ab, div-1, r) +
					draw_triangles( c, ac, bc, div-1, r) +
					draw_triangles(ab, bc, ac, div-1, r);
			}
		}

		static void draw(int num_subdiv, float radius = 1.0f) {
			unsigned num_tris = 0;
			glBegin(GL_TRIANGLES);
				for (unsigned i=0; i<20; ++i)
					num_tris += draw_triangles(vdata[tindices[i][0]], vdata[tindices[i][1]], vdata[tindices[i][2]], num_subdiv, radius);
			glEnd();
			printf("geosphere %d triangles\n", num_tris);
		}
	}

	static void process(const float time_abs, const float time_delta) {
		params.is_progressive = true;
		params.verbose = false;

		if (gl::state::tex_and_pbo)
			last_render_time = gl::state::tex_and_pbo ? gl::state::tex_and_pbo->render(&params) : 0;
		++params.pass;
	}

	// assumes blending off, depth_test on.
	// on exit blending on.
	static void draw_scene_outline(bool only_selected, float time_abs) {
		static GLuint l_solid_sphere = -1;
		if (only_selected && !scene.is_valid_selection())
			return; // nothing to draw

		const uint8_t alpha = 48; // 32 + cyclical_frac(time_abs/2)*16
		const uint8_t
			color_unselected[4] = {   255, 255, 255,  32 },
			color_selected[2][4]= {	{ 255, 191,   0,  64 }, // make it stand out more
									{ 192, 220, 192,  alpha } };
		enum { slices = 32, stacks = 32 };
		const float factor = 1, units = 2;

		if (l_solid_sphere == -1u) {
			l_solid_sphere = glGenLists(1);
			glNewList(l_solid_sphere, GL_COMPILE);
				if (1) {
					glRotatef(180.f/slices, 0, 1, 0);	// disalign a bit (vs grid).
					glRotatef(90, 1, 0, 0);				// world_up anyone?
					glutSolidSphere(1, slices, stacks);
				}
				else
					// doesn't look as good.
					geosphere::draw(5, 1);
			glEndList();
		}
		glPolygonOffset(factor, units); // push the solid, because you can't push lines.
		glLineWidth(.75);

		// first pass to set the depth buffer, second to draw.
		for (unsigned pass=0; pass<2; ++pass) {
			if (pass == 0) { // depth write pass
				glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
				glDepthMask(GL_TRUE);
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				glEnable(GL_POLYGON_OFFSET_FILL);
			}
			else {
				glEnable(GL_BLEND);
				glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
				glDepthMask(GL_FALSE);
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				glDisable(GL_POLYGON_OFFSET_FILL);
			}

			for (unsigned i=0; i<params.num_spheres; ++i) {
				if (only_selected && pass > 0 && i != scene.get_selection_id())
					continue;
				bool_t is_selected = (pass > 0) && (i == scene.get_selection_id());
				glColor4ubv(is_selected ? color_selected[only_selected] : color_unselected);
				glPushMatrix();
					const sphere_t &s(scene.get(i));
					float radius = s.rad;
					glTranslatef(s.pos[0], s.pos[1], s.pos[2]);
					glScalef(radius, radius, radius);
					glCallList(l_solid_sphere);
				glPopMatrix();
			}
		}
		// blend on, depth mask off, front&back line. leave blend on
		glDepthMask(GL_TRUE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}


	static void draw_panel(unsigned id, const sphere_t &sphere) {
		#define V(prefix, vec) sys::fmt_t(prefix " (%6.3f,%6.3f,%6.3f)", vec.x(), vec.y(), vec.z())
		sys::fmt_t lines[] = {
			sys::fmt_t("sphere #%d type %s", id, ::misc::refl2string(refl_t(sphere.type))),
			sys::fmt_t("radius %9f", sphere.rad),
			V("pos", sphere.pos),
			V("emi", sphere.emi),
			V("col", sphere.col)
		};
		#undef V
		int w = 0;
		enum { num_lines = sizeof(lines)/sizeof(lines[0]) };
		for (unsigned i=0; i<num_lines; ++i)
			w = std::max(w, misc::stroke::width(lines[i]));

		// line height
		vec_t l(0, misc::stroke::glyph_h, 0);

		glTranslatef(-w/2.f, 0, 0); // center
		for (unsigned i=num_lines; i; --i) {
			glPushMatrix();
				misc::stroke::print(lines[i-1]);
			glPopMatrix();
			glTranslatef(l.x(), l.y(), l.z()); // next line
		}
	}

	// simple spherical billboarding, because that's exactly what we need.
	static mat4_t billboard_spherical(const mat4_t &mdv) {
		return mat4_t::from_axis(
			vec_t(1, 0, 0),
			vec_t(0, 1, 0),
			vec_t(0, 0, 1),
			vec_t(mdv[3][0], mdv[3][1], mdv[3][2]) );
	}

	// assumes blending on.
	static void draw_panels(bool only_selected, const mat4_t &mdv) {
		if (only_selected && !scene.is_valid_selection())
			return; // nothing to draw

		const mat4_t mat_bb(billboard_spherical(mdv));
		// those stroke font are really large.
		// and i don't know what would be the proper way to adjust dynamically.
		// (don't want to clog the view etc... for now best heuristic is no heuristic).
		const float
			unit_glyph = 1.f/misc::stroke::glyph_h,
			adjust = 48*2, // 4*2
			scale = unit_glyph/adjust,
			offset = 5 /*lines or so*/  / (2*adjust);


		glColor4ub(255, 255, 255, 128);
		glLineWidth(.5f); // looks like crap when not antialiased enough.
		for (unsigned i=0; i<params.num_spheres; ++i) {
			if (only_selected && i != scene.get_selection_id())
				continue;
			const sphere_t &sphere(scene.get(i));
			// where are we gonna put that... hmm..
			vec_t pos(sphere.pos + -gl::state::smp.dy.norm()*(sphere.rad + offset));
			vec_t tpos(mdv*pos);
			glPushMatrix();
				glLoadMatrixf(mat_bb.get());
				glTranslatef(tpos[0], tpos[1], tpos[2]);
				glScalef(scale, scale, scale);

				draw_panel(i, sphere);
			glPopMatrix();
		}
	}

	static void display(const float time_abs, const float time_delta) {
		static GLuint l_quad_setup_draw = -1;
		glViewport(0, 0, current_width, current_height);

		if (l_quad_setup_draw == -1u) {
			l_quad_setup_draw = glGenLists(1);
			glNewList(l_quad_setup_draw, GL_COMPILE);
				glMatrixMode(GL_PROJECTION);
				glPushMatrix();
				glLoadIdentity();
				glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
				glMatrixMode(GL_MODELVIEW);
				glLoadIdentity();

				glDisable(GL_DEPTH_TEST);
				glEnable(GL_TEXTURE_2D);
				glColor4ub(255, 255, 255, 255);
				glBegin(GL_QUADS);
					// if we've flipped our screen vertically, unflip here.
					const float
						z = .5f,
						v[4][3]  = { { -1,-1, z }, { +1,-1, z }, { +1,+1, z }, { -1,+1, z } },
						tc[4][2] = {     { 0, 0 },     { 1, 0 },     { 1, 1 },     { 0, 1 } };
					const int ti[2][4] = { { 0,1,2,3 }, { 3,2,1,0 } };

					glTexCoord2fv(tc[ti[gl::camera_t::flip_y][0]]); glVertex3fv(v[0]);
					glTexCoord2fv(tc[ti[gl::camera_t::flip_y][1]]); glVertex3fv(v[1]);
					glTexCoord2fv(tc[ti[gl::camera_t::flip_y][2]]); glVertex3fv(v[2]);
					glTexCoord2fv(tc[ti[gl::camera_t::flip_y][3]]); glVertex3fv(v[3]);
				glEnd();
				glMatrixMode(GL_PROJECTION);
				glPopMatrix();
				glDisable(GL_TEXTURE_2D);

				// also, prepare some more states.
				glEnable(GL_DEPTH_TEST);
			glEndList();
		}
		else
			glCallList(l_quad_setup_draw);

		//
		// match the view from the raytracer,
		// and augment it a bit.
		//
		// leave it on, doesn't make any difference speed wise.
		// if (state::use_multisamp) glEnable (GL_MULTISAMPLE_ARB);
		const float aspect_ratio = float(current_width)/float(current_height); // t_far = 2*16, t_near = .0125f /* 1 */;
		float t_near = 1, t_far = 1;
		scene.estimate_near_far(gl::state::cam.get_eye(), t_near, t_far);
		t_near /= 2; // heavy handed.

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gl::state::cam.set_frustum(aspect_ratio, t_near, t_far);
		mat4_t mdv = gl::state::cam.make_view();
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(mdv.get());

		draw_scene_outline(!show_ui, time_abs);
		// blending is on

		if (show_ui) {
			// glColor4ub(255, 255, 64, 48); mimi
			glColor4ub(255, 255, 255, 24);
			glLineWidth(.25f);
			// draw various crap
			misc::draw_grid(10, 1);
		}
		// needs to draw info at least for the selection.
		glEnable(GL_LINE_SMOOTH);
		draw_panels(!show_ui, mdv);
		glDisable(GL_LINE_SMOOTH);

		glDisable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);


		//if (state::use_multisamp) glDisable (GL_MULTISAMPLE_ARB);

		// for the rest, that is 2D stuff, we'll let nvui set everything up.
		// and we'll hook back in.
		CUT_CHECK_ERROR_GL();
	}

	// called with nvui bindings.
	static void display_post_nvui_hook(const float time_abs, const float time_delta) {
		// yay.
	}

	// per frame work.
	static void go() {
		static float last_time = -1;
		float t = cutGetTimerValue(timer)/1000.f;
		float dt = last_time > -1 ? t - last_time : 0;
		last_time = t;

		// FreeGlut postpones resizing when starting, wait until it settles in.
		if (gl::current_height == 0 && gl::current_width == 0) return;

		const point_t res(current_width, current_height);
		if (texture_and_pbo_t::resize_everything_if(res, renderer, gl::state::tex_and_pbo))
			smallpt_reset_accumulator(); // be sure to wipe it all.

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		ui::update(t, dt, gl::state::cam);
		if (gl::state::is_camera_dirty()) {
			assert(gl::state::tex_and_pbo && "pas de bras, pas de chocolat");
			gl::state::smp = gl::state::cam.make_sampler(gl::state::tex_and_pbo->res); // if that's what's mapped.
			const vec_t cu_cam[4] = {
				gl::state::smp.dx, gl::state::smp.dy,
				gl::state::smp.corner, gl::state::cam.get_eye()
			};
			cuda_upload_camera(cu_cam);
			gl::state::copy_camera();
		}
		scene.upload_if(params);

		process(t, dt);
		display(t, dt);
		ui::display(t, dt);
	}

	// res is the intitial rendering resolution.
	static void init(const point_t res, const char * const scene_filename) {
		gl::timer = 0;
		cutCreateTimer(&gl::timer);
		cutStartTimer(gl::timer);

		gl::misc::stroke::init();

		// glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
		// glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
		// glEnable(GL_LINE_SMOOTH);

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

		// glDisable (GL_MULTISAMPLE_ARB);
		glEnable (GL_MULTISAMPLE_ARB);
		// glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_FASTEST);
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
		glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);

		// prime the scene data and camera.
		scene.init(gl::state::cam, scene_filename);
		//
		// rest of init will happen naturally as a resize.
		//
	}
}



// ===========================================================================
//							UI
// ===========================================================================

// for now, be sure to pull our patched version.
#include "nv/nvGlutWidgets.h"

//
// note:
//   update will be called first, then display.

namespace ui {
	namespace details {
		struct toggles_t {
			uint32_t bits;
			toggles_t() : bits(0) {}
			template<unsigned idx> void set(bool_t on) {
				uint32_t mask = 1u<<idx;
				bits |= mask;
				bits ^= on ? 0 : mask;
			}
			template<unsigned idx> int get() const { return bits & (1u<<idx) ? 1 : 0; }
		};
		namespace toggles {
			enum { // bit names
				dummy = 0,
				mod_ctrl, mod_alt, mod_shift,
				move_l , move_r, move_f, move_b, move_u, move_d // movement on 3 axis.
			};
		}
		union kdb_event_t {
			uint32_t bits;
			struct {
				uint32_t key : 8;
				uint32_t is_special : 1;
				uint32_t is_down : 1;
				uint32_t mod_shift : 1;
				uint32_t mod_ctrl : 1;
				uint32_t mod_alt : 1;
			};
		};
	}
	// need to record set/reset actions or we'd miss set/reset happening within one frame.
	// tg[0] is the set set, tg[1] the reset set.
	static details::toggles_t toggles[2];

	static nv::GlutUIContext nvui;
	// persistent ui states.
	namespace state {
		static bool unfold_params = false, unfold_sphere = true;

		static bool unfold_help = false;

		//FIXME: overkill and unused, remove cruft.
		struct pseudo_menu_t {
			union unfold_t {
				uint32_t bits;
				struct {
					uint32_t root : 1;
					uint32_t opt1 : 1;
					uint32_t opt2 : 1;
					uint32_t opt3 : 1;
				};
			} unfold;
			uint32_t prev_bits;

			bool_t begin() const { return unfold.bits != 0; }
			void end() { prev_bits = unfold.bits; }
			unfold_t edge() const {
				unfold_t u = { unfold.bits };
				u.bits ^= prev_bits;
				return u;
			}
		};
		static pseudo_menu_t pseudo_menu;

		// because other actions need to be modulated, that speed is cached here.
		// camera_modulation, if + accelerates, - decelerates.
		static float camera_cached_speed = 1, camera_modulation = 0;
	}


	static void init() {
		nvui.init();
	}

	// called from callbacks.
	static void kbd(const details::kdb_event_t e, const point_t coords) {
		static int watchdog_scene_wipe = 0;
		--watchdog_scene_wipe;
		// glut gets modifiers wrong (ie junk on release). kludge.
		// toggles[e.mod_shift ? 0 : 1].set<details::toggles::mod_shift>(true);
		toggles[0].set<details::toggles::mod_shift>(e.mod_shift);
		toggles[0].set<details::toggles::mod_ctrl>(e.mod_ctrl);
		toggles[0].set<details::toggles::mod_alt>(e.mod_alt);

		const unsigned tg_idx = e.is_down ? 0 : 1;
		enum { smask = 512u };
		const unsigned key = e.key|(e.is_special ? smask : 0);
		if (e.is_down) switch(key) { // key press only.
			case 'q':
			case  27: exit(0);

			case 'z': return smallpt_reset_accumulator();	// reset accumulator.
			case 'u': gl::show_ui = !gl::show_ui; return;	// hide/show full ui.
			case 'p': state::unfold_params = !state::unfold_params; return;
			case 's': state::unfold_sphere = !state::unfold_sphere; return;

			// case 'm': gl::state::use_multisamp = !gl::state::use_multisamp; printf("gl::state::use_multisamp %d\n",gl::state::use_multisamp); return;
			case 'w': // scene wipe. need to hit w twice in succession.
				if (watchdog_scene_wipe == 0) scene.clear();
				else watchdog_scene_wipe = 2; // release + press.
				return;

			case 'x': // cycle world's up
				gl::state::cam.cycle_world_up();
				//FIXME: such output/feedback should be displayed somehow.
				printf("world up(%+g, %+g, %+g)\n", gl::state::cam.get_world_up().x(), gl::state::cam.get_world_up().y(), gl::state::cam.get_world_up().z());
				return;


			case '\b': scene.deselect(); return; // deselect.
			case '+':
			case '-': scene.cycle_selection(key == '+' ? +1 : -1); return;	// cycle selection.

			case 0x03: if (e.mod_ctrl) scene.copy(); return;	// ctrl-c
			case 0x16: // ctrl-v (same place or in front of camera with shift)
				if (e.mod_ctrl) {
					if (e.mod_shift)
						scene.paste(gl::state::cam.get_eye() + gl::state::cam.get_fwd());
					else
						scene.paste();
				}
				return;

			case 0x7f: scene.del(); return; // delete selection, del.


			case 'h': state::unfold_help = !state::unfold_help; return;

			case smask|GLUT_KEY_F1: state::unfold_help = !state::unfold_help; return;
			case smask|GLUT_KEY_F2: state::pseudo_menu.unfold.root = !state::pseudo_menu.unfold.root; return;

			// load/save.
			case smask|GLUT_KEY_F5 : scene.save(gl::state::cam); return;
			case smask|GLUT_KEY_F9 : scene.load(gl::state::cam); smallpt_reset_accumulator(); return;
			case smask|GLUT_KEY_F10: scene.load_old(gl::state::cam); smallpt_reset_accumulator(); return; // transitional.

			case smask|GLUT_KEY_HOME: // look
				if (scene.is_valid_selection()) gl::state::cam.look_at(scene.selection().pos);
				return;
		}

		switch(key) { // press+release.
			// 'special' keys
			// movement
			case smask|GLUT_KEY_LEFT:		toggles[tg_idx].set<details::toggles::move_l>(true); return;
			case smask|GLUT_KEY_UP:			toggles[tg_idx].set<details::toggles::move_f>(true); return;
			case smask|GLUT_KEY_RIGHT:		toggles[tg_idx].set<details::toggles::move_r>(true); return;
			case smask|GLUT_KEY_DOWN:		toggles[tg_idx].set<details::toggles::move_b>(true); return;
			case smask|GLUT_KEY_PAGE_UP:	toggles[tg_idx].set<details::toggles::move_u>(true); return;
			case smask|GLUT_KEY_PAGE_DOWN:	toggles[tg_idx].set<details::toggles::move_d>(true); return;
		}
	}


	static void update(const float time_abs, const float time_delta, gl::camera_t &camera) {
		const float delta_mod_step = 8;
		float delta_mod = 1;
		if (toggles[0].get<details::toggles::mod_shift>()) delta_mod /= delta_mod_step;
		if (toggles[0].get<details::toggles::mod_ctrl>()) delta_mod *= delta_mod_step;
		const float delta = time_delta*delta_mod;

		// movement.
		float cam_speed = scene.estimate_max_extent()/16; // 16s to zip through scene @ base speed.
		// but that heuristic may stinks, have a UI knob to modulate.
		// cam_speed *= state::camera_modulation >= 0 ? 1 + state::camera_modulation : -1/state::camera_modulation;
		cam_speed *= state::camera_modulation >= 0 ? math::square(1 + state::camera_modulation) : 1/math::square(-1+state::camera_modulation);
		state::camera_cached_speed = cam_speed;

		float d[3] = {
			-toggles[0].get<details::toggles::move_l>() + toggles[0].get<details::toggles::move_r>(),
			-toggles[0].get<details::toggles::move_d>() + toggles[0].get<details::toggles::move_u>(),
			-toggles[0].get<details::toggles::move_b>() + toggles[0].get<details::toggles::move_f>()
		};
		for (unsigned i=0; i<3; ++i) d[i] *= delta*cam_speed;
		camera.set_eye(camera.get_eye() + camera.get_lft()*d[0] + camera.get_up()*d[1] + camera.get_fwd()*d[2]);

		// now update those toggles.
		toggles[0].bits ^= toggles[1].bits;
		toggles[1].bits = 0;
	}

	//
	// mouse handling.
	//
	// we pull mouse events from nvui:
	// . because we have to guess if that event was already used by the ui (then we're not called)
	// . it does edge detection and other bookkeeping
	// note that it flips the y coord, and nv::ButtonState records begin/end states (actual coords come from getCursor*).
	namespace details {
		static point_t convert(const nv::Point &p) { return point_t(p.x, gl::current_height-p.y); }
		static point_t cursor() { return point_t(nvui.getCursorX(), gl::current_height-nvui.getCursorY()); }
		static float signed_distance(const point_t p) {
			float s = std::abs(p.x) > std::abs(p.y) ? p.x : p.y;
			float d = math::sqrt(p.x*p.x + p.y*p.y);
			return s > 0 ? d : -d; // no copysignf
		}
	}
	static void handle_mouse() {
		static point_t last_coords(0, 0);
		const point_t coords(details::cursor()); // mouse coords, right now.
		for (unsigned button=0; button<3; ++button) {
			enum { mod_mask = nv::ButtonFlags_Shift | nv::ButtonFlags_Ctrl | nv::ButtonFlags_Alt };
			const int state = nvui.getMouseState(button).state;
			if (0) {
				char S[7] = "OBEsca";
				if (!(state & nv::ButtonFlags_On))    S[0] = '.';
				if (!(state & nv::ButtonFlags_Begin)) S[1] = '.';
				if (!(state & nv::ButtonFlags_End))   S[2] = '.';
				if (!(state & nv::ButtonFlags_Shift)) S[3] = '.';
				if (!(state & nv::ButtonFlags_Ctrl))  S[4] = '.';
				if (!(state & nv::ButtonFlags_Alt))   S[5] = '.';
				glRasterPos2i(gl::current_width-64, /*gl::current_height -*/ gl::misc::bitmap::line_height*(button+1));
				gl::misc::bitmap::print(S);
			}


			if (!(state & nv::ButtonFlags_On)) continue; // nothing.

			// Begin cannot be reliably detected.
			const bool is_begin = state & nv::ButtonFlags_Begin ? true : false, is_end = state & nv::ButtonFlags_End ? true : false, is_motion = !(is_begin|is_end);
			const point_t edge_coords(details::convert(nvui.getMouseState(button).cursor)); // and where it was.
			const point_t d_coords(is_begin ? point_t(0, 0) : coords - last_coords);
			switch (button)  {
				case 0: // left mouse button; change view direction or spawn spheres.
					if (is_end) {
						if ((state & nv::ButtonFlags_Alt)) {
							float d = 1;
							scene.addd(gl::state::cam.get_eye() + gl::state::smp.map(coords)*d);
						}
						else
							gl::state::cam.look_at(gl::state::cam.get_eye() + gl::state::smp.map(coords));
						smallpt_reset_accumulator();
					}
					break;
				case 1: // middle mouse button; moves camera (shift = other plane)
					if (is_motion|is_end)
						if (d_coords != point_t(0, 0)) {
							// float dx = float(d_coords.x)/gl::current_width, dy = float(d_coords.y)/gl::current_height;
							float dx = state::camera_cached_speed*d_coords.x/gl::current_width;
							float dy = state::camera_cached_speed*d_coords.y/gl::current_height;
							if (state & nv::ButtonFlags_Shift)
								gl::state::cam.set_eye(gl::state::cam.get_eye() + gl::state::cam.get_lft()*dx - gl::state::cam.get_fwd()*dy);
							else
								gl::state::cam.set_eye(gl::state::cam.get_eye() + gl::state::cam.get_lft()*dx - gl::state::cam.get_up()*dy);
							smallpt_reset_accumulator();
						}
					break;
				case 2:
					{
						bool dirty = false;
						switch (state & mod_mask) {
							case 0: // no modifier, picking
								if (is_begin)
									scene.picking(gl::state::cam.get_eye(), gl::state::smp.map(edge_coords).norm());
								break;
							case nv::ButtonFlags_Ctrl: // incrementally modify its radius (don't store any state)
								if (scene.is_valid_selection()) {
									scene.selection().rad *= 1 + clamp(-.75f, .75f, details::signed_distance(d_coords)/64);
									dirty = true;
								}
								break;
							case nv::ButtonFlags_Alt: // move sphere (shift = other plane)
							case nv::ButtonFlags_Alt | nv::ButtonFlags_Shift:
								if (scene.is_valid_selection()) {
									sphere_t &s(scene.selection());
									float dx = state::camera_cached_speed*d_coords.x/gl::current_width;
									float dy = state::camera_cached_speed*d_coords.y/gl::current_height;
									if (state & nv::ButtonFlags_Shift)
										s.pos = s.pos + gl::state::cam.get_lft()*dx - gl::state::cam.get_fwd()*dy;
									else
										s.pos = s.pos + gl::state::cam.get_lft()*dx + gl::state::cam.get_up()*dy;
									dirty = true;
								}
								break;
						}
						if (dirty) {
							scene.set_dirty();
							smallpt_reset_accumulator();
						}
					}
					break;
			}
		}
		last_coords = coords;
	}

	enum {
		// a group flag to put widgets on a line.
		gf_one_line =  nv::GroupFlags_LayoutHorizontal | nv::GroupFlags_LayoutNoMargin | nv::GroupFlags_AlignRight // | GroupFlags_LayoutNoSpace
	};
	static nv::Rect rnone;


	// sadly, we have to hold a state of the scalar/vector we're attached to,
	// to avoid a per frame re-evaluation mayhem; but that means we may be out of sync;
	// and nvui doesn't provide a way to detect when that input field is 'validated'. Hmm.
	// Well, it appears it's good enough not to float->string->float but simply skip the first step, because
	// our float-string evaluates to the same float (most of the time).
	struct numeric_field_t {
		enum types_t { T_BOGUS, T_SCALAR = 1, T_VECTOR = 3 };
		struct float_buffer_t {
			enum { storage_size = 63 };
			char storage[storage_size+1];
			void set(const float &f) { snprintf(storage, storage_size, "%g", f); storage[storage_size] = 0; }
			bool_t evaluate(float &dst, const int len) {
				// ensure dst is unharmed; simpler to acheive with sscanf.
				storage[len] = 0;
				float dummy;
				if (std::sscanf(storage, "%f", &dummy) != 1)
					return false;
				dst = dummy;
				return true;
			}
		};

		void *value;
		float_buffer_t buffers[3];	// we need to hold values in string form.
		float original[3];			// to detect out of sync.
		types_t type;
		bool out_of_sync;
		// nv::Rect rect; // right now, it's rather useless to try to get them placed; fix nvui.

		numeric_field_t() : value(0), type(T_BOGUS) {}
		numeric_field_t(float *scalar) : value(scalar), type(T_SCALAR), out_of_sync(true) {}
		numeric_field_t(vec_t *vec) : value(vec), type(T_VECTOR), out_of_sync(true) {}

		template<typename T> bool_t is_same(T *p) const { return value == p; }

		void run() {
			nvui.beginGroup(nv::GroupFlags_GrowUpFromRight); // nv::GroupFlags_LayoutDefault
			nvui.beginFrame();
				nvui.beginGroup(gf_one_line, rnone);
					float * const p = static_cast<float*>(value);
					// first, detect if we're out of sync.
					for (int i=0; i<type; ++i) out_of_sync |= p[i] != original[i];
					if (out_of_sync) { // reload
						for (int i=0; i<type; ++i) original[i] = p[i];
						for (int i=0; i<type; ++i) buffers[i].set(original[i]);
						out_of_sync = false;
					}
					// grr, they are placed in reverse order.
					for (int i=type-1; i>=0; --i) {
						int buf_len = 0;
						if (nvui.doLineEdit(rnone, buffers[i].storage, float_buffer_t::storage_size, &buf_len, 0)) {
							if (buffers[i].evaluate(p[i], buf_len))
								scene.set_dirty(); //FIXME: not always.
						}
					}
				nvui.endGroup();
			nvui.endFrame();
			nvui.endGroup();
		}
	};

	static numeric_field_t *numeric_field = 0;
	// a) display a clickable button b) attach or detach to a value behind it, whether that button gets clicked.
	template<typename T>
		static void editable(const nv::Rect &r, const char * const txt, T *value) {
			static numeric_field_t store;
			if (nvui.doButton(r, txt, 0)) {
				if (numeric_field && numeric_field->is_same(value)) // already set, remove
					numeric_field = 0;
				else {
					new (&store) numeric_field_t(value);
					numeric_field = &store;
				}
			}
		}

	static bool nvui_wants_kbd = false;

	// we need to filter keys nvui sees, because of its behavior vs weird input.
	template<bool is_special>
		static void submit_key(uint8_t key, int x, int y) {
			if (!is_special) switch (key) {
				case '\r': case '\b': case '.': case 'e':
				case '0': case '1': case '2': case '3':
				case '4': case '5': case '6': case '7':
				case '8': case '9': case '+': case '-':
				case 127: // delete
					nvui.keyboard(key, x,y);
				default:
					return;
			}
			else switch (key) {
				case GLUT_KEY_LEFT: case GLUT_KEY_RIGHT:
				case GLUT_KEY_HOME: case GLUT_KEY_END:
					nvui.specialKeyboard(key, x, y);
				default:
					return;
			}
		}

	static void display(const float time_abs, const float time_delta) {
		using namespace nv;

		enum { nvui_margin = 5 };
		bool focused = false; // if true, we should ignore clicks and motion.
		bool dirty = false;

		nvui.begin();
			//
			// Params
			//
			// gamma, fovy, max_paths, spp
			nvui.beginGroup(GroupFlags_GrowDownFromLeft);
				nvui.beginGroup(gf_one_line);
					nvui.doButton(rnone, "P", &state::unfold_params, 0);
					if (state::unfold_params) {
						// some silly stats
						nvui.doLabel(rnone, sys::fmt_t("fps %5.2f", 1/time_delta), 1);
						const float
							rate = 1/(gl::last_render_time/1000),
							spp_per_mn  = 60*(params.spp)*rate;
						nvui.doLabel(rnone, sys::fmt_t("spp %5d, %5.0f/mn", params.pass*params.spp, spp_per_mn), 1);
					}
				nvui.endGroup();
				if (state::unfold_params) {
					enum { style_lbl = 1 };
					float dummy;
					const nv::Rect rlabel(0,0, 82, 0);
					nvui.beginGroup(gf_one_line);
						// gamma
						editable(rlabel, sys::fmt_t("gamma %6.2f", params.gamma), &params.gamma);
						nvui.doHorizontalSlider(rnone, 0.0125f, 8.f, &params.gamma);
					nvui.endGroup();
					nvui.beginGroup(gf_one_line);
						// fovy
						editable(rlabel, sys::fmt_t("fovy    %6.2f", gl::state::cam.fovy), &gl::state::cam.fovy);
						nvui.doHorizontalSlider(rnone, 0, 120.f, &gl::state::cam.fovy);
						gl::state::cam.fovy = gl::state::cam.fovy <= 0 ? 0.125f : gl::state::cam.fovy;
					nvui.endGroup();
					nvui.beginGroup(gf_one_line);
						// spp per pass
						dummy = params.spp/(SS*SS);
						nvui.doLabel(rlabel, sys::fmt_t("spppp       %d", params.spp), style_lbl);
						nvui.doHorizontalSlider(rnone, 1, 32, &dummy);
						params.spp = dummy > 1 ? int(dummy)*(SS*SS) : SS*SS;
					nvui.endGroup();
					nvui.beginGroup(gf_one_line);
						// max paths
						dummy = params.max_paths;
						nvui.doLabel(rlabel, sys::fmt_t("max paths %d", params.max_paths), style_lbl);
						nvui.doHorizontalSlider(rnone, 1, 64, &dummy);
						params.max_paths = dummy > 1 ? int(dummy) : 1;
					nvui.endGroup();
					// camera_modulation
					nvui.beginGroup(gf_one_line);
						// camera_modulation
						editable(rlabel, sys::fmt_t("cam     %+5.2f", state::camera_modulation), &state::camera_modulation);
						nvui.doHorizontalSlider(rnone, -4, +4, &state::camera_modulation);
					nvui.endGroup();
					nvui.beginGroup(gf_one_line);
						if (nvui.doButton(rlabel, "reset acc"))
							smallpt_reset_accumulator();
					nvui.endGroup();
				}
				focused |= nvui.isOnFocus();
			nvui.endGroup();

			//
			// sphere
			//
			// emission, color, radius, type.
			nvui.beginGroup(GroupFlags_GrowDownFromRight);
			if (!(state::unfold_sphere && scene.is_valid_selection()))
				nvui.doButton(rnone, "S", &state::unfold_sphere, 0); // a little button.
			else {
				enum {
					label_w = 28,
					sld_w = 72, sld3_w = 3*sld_w + 2*nvui_margin,
					max_w = sld3_w + nvui_margin + label_w,
					top_w = (max_w-2*nvui_margin) / 3
				};
				static const nv::Rect rlabel(0,0,label_w,0), rsld(0,0,sld_w), rsld3(0, 0, sld3_w, 0);
				static const char *types[3] = { "diffuse", "specular", "refractive" };

				sphere_t &s(scene.selection());

				nvui.beginGroup(gf_one_line); // top 3 buttons.
				{
					const nv::Rect rtop(0, 0, top_w, 0);
					nvui.doButton(rtop, sys::fmt_t("Sphere #%2d", scene.get_selection_id()), &state::unfold_sphere, 0);
					editable(rtop, "position", &s.pos); // position
					// type
					dirty |= nvui.doComboBox(rtop, 3, types, &s.type, 0);
				}
				nvui.endGroup();

				// color
				nvui.beginGroup(gf_one_line);
				{
					const float fl = 0, fh = 1;
					vec_t &v(s.col);
					editable(rlabel, "col", &v);
					dirty |= nvui.doHorizontalSlider(rsld, fl, fh, &v.m[2]);
					dirty |= nvui.doHorizontalSlider(rsld, fl, fh, &v.m[1]);
					dirty |= nvui.doHorizontalSlider(rsld, fl, fh, &v.m[0]);
				}
				nvui.endGroup();
				// emission
				nvui.beginGroup(gf_one_line);
				{
					const float fl = 0, fh = 1;
					vec_t &v(s.emi);
					editable(rlabel, "emi", &v);
					dirty |= nvui.doHorizontalSlider(rsld, fl, fh, &v.m[2]);
					dirty |= nvui.doHorizontalSlider(rsld, fl, fh, &v.m[1]);
					dirty |= nvui.doHorizontalSlider(rsld, fl, fh, &v.m[0]);
				}
				nvui.endGroup();
				// radius
				nvui.beginGroup(gf_one_line);
				{
					const float fl = 0, fh = 4;
					editable(rlabel, "rad", &s.rad);
					dirty |= nvui.doHorizontalSlider(rsld3, fl, fh, &s.rad);
				}
				nvui.endGroup();
			}
				focused |= nvui.isOnFocus();
			nvui.endGroup();

			//
			// help
			//
			if (state::unfold_help) {
				const char txt[] =
					"That's supposed to be some help text.\n"
					"\n"
					"nvui.beginGroup(gf_one_line);\n"
					"{\n"
					"	const float fl = 0, fh = 1;\n"
					"	vec_t &v(s.col);\n"
					"	editable(rlabel, \"col\", &v);\n"
					"	dirty |= nvui.doHorizontalSlider(rsld, fl, fh, &v.m[2]);\n"
					"	dirty |= nvui.doHorizontalSlider(rsld, fl, fh, &v.m[1]);\n"
					"	dirty |= nvui.doHorizontalSlider(rsld, fl, fh, &v.m[0]);\n"
					"}\n"
					"nvui.endGroup();\n";

				nvui.doLabel(rnone, txt, 1);
			}

			//
			// pseudo menu handling, for global operations.
			//
			if (state::pseudo_menu.begin()) {
				enum {
					vlbl_w = 64, vsld_w = 96, vedi_w = 64, // typical 3 widget line.
					total_w = vlbl_w + vsld_w + vlbl_w + 2*nvui_margin,
					// half_w = (total_w - nvui_margin)/2
					half_w = (total_w + (2-1)*nvui_margin)/2 // grr, they are out of panels, hence more margin.
				};
				static nv::Rect menu_rect(256, 256, 0, 0);

				const state::pseudo_menu_t::unfold_t edge = state::pseudo_menu.edge();
				bool unfold_menu = true;
				nvui.beginPanel(menu_rect, "global operations", &unfold_menu);
				if (unfold_menu) {
					static const nv::Rect
						rlbl(0,0,vlbl_w,0), rsld(0,0,vsld_w, 0), redi(0, 0, vedi_w, 0),
						rhalf(0, 0, half_w, 0), rline(0, 0, total_w, 0);
					static bool unfold_menus[2] = { 0 };
					unsigned menu_idx = 0;
					// refit
					static float refit_max_dim = 1;
					static bool refit_fit = true;
					// englobe
					static float englobe_space = 0, englobe_scale = 1;
					static bool englobe_hollow = false;
					nvui.beginGroup();
						//
						// scene refit
						//
						nvui.beginPanel(rnone, "refit", &unfold_menus[menu_idx]);
						if (unfold_menus[menu_idx]) {
							if (nvui.doButton(rline, "shrink that scene, now!", 0, 1))
								scene.mdl_refit(gl::state::cam, refit_fit, refit_max_dim);
							nvui.doCheckButton(rnone, "fit", &refit_fit);
							nvui.beginGroup(gf_one_line);
								nvui.doLabel(rlbl, refit_fit ? "max dim" : "1/scale"); nvui.doHorizontalSlider(rsld, 0, 4, &refit_max_dim); editable(redi, sys::fmt_t("%1.3f", refit_max_dim), &refit_max_dim);
							nvui.endGroup();
						}
						nvui.endPanel();
						++menu_idx;
						//
						// scene boxing
						//
						nvui.beginPanel(rnone, "englobe", &unfold_menus[menu_idx]);
						if (unfold_menus[menu_idx]) {
							if (nvui.doButton(rline, englobe_hollow ? "yeah, box it!" : "wrap a sphere around it, please.", 0, 1))
								scene.mdl_englobe(englobe_hollow, englobe_space, englobe_scale);
							nvui.doCheckButton(rnone, "hollow", &englobe_hollow);
							nvui.beginGroup(gf_one_line);
								nvui.doLabel(rlbl, "space"); nvui.doHorizontalSlider(rsld, 0, 4, &englobe_space); editable(redi, sys::fmt_t("%1.3f", englobe_space), &englobe_space);
							nvui.endGroup();
							nvui.beginGroup(gf_one_line);
								nvui.doLabel(rlbl, "scale"); nvui.doHorizontalSlider(rsld, 0, 4, &englobe_scale); editable(redi, sys::fmt_t("%1.3f", englobe_scale), &englobe_scale);
							nvui.endGroup();
						}
						nvui.endPanel();
						++menu_idx;
						//
						// save / load
						//
						nvui.beginGroup(gf_one_line);
							if (nvui.doButton(rhalf, "load"))
								if (scene.load(gl::state::cam))
									smallpt_reset_accumulator();
							if (nvui.doButton(rhalf, "save"))
								scene.save(gl::state::cam);
						nvui.endGroup();
					nvui.endGroup();
				}
				else
					state::pseudo_menu.unfold.bits = 0;

				focused |= nvui.isOnFocus();
				if (1 && edge.bits) {
					// try to center
					int w = nvui.getGroupWidth(), h = nvui.getGroupHeight();
					menu_rect.x = (gl::current_width-w)/2;
					menu_rect.y = (gl::current_height-h)/2;
					menu_rect.y += h; // but i give up trying to understand how panels are sized/placed.
				}
				nvui.endPanel();
			}
			state::pseudo_menu.end();


			// we're almost done now.
			if (dirty)
				scene.set_dirty();

			// maybe add some widgets to tweak some values.
			if (numeric_field)
				numeric_field->run();
			focused |= nvui.isOnFocus();

			// direct everything to nvui?
			nvui_wants_kbd = focused;

			// do something with the mouse if it wasn't absorbed.
			if (!focused)
				handle_mouse();

			// if we have to draw some more 2D stuff,
			// use the transfo nvui has set
			gl::display_post_nvui_hook(time_abs, time_delta);
		nvui.end();
	}
}

// ===========================================================================
//							GLUT
// ===========================================================================

void display() {
	gl::go();
	glutSwapBuffers();
}

void reshape(int new_w, int new_h) {
	static int W = glutGet(GLUT_SCREEN_WIDTH), H = glutGet(GLUT_SCREEN_HEIGHT);
	if (new_w <= 0 || new_h <= 0) return; // minimization. hmpff.
	int
		w = misc::argument_reduction(W, int(tile_size_x), tweaks::make_res_x(new_w)),
		h = misc::argument_reduction(H, int(tile_size_y), tweaks::make_res_y(new_h));
	// printf("reshape: (%d, %d) -> (%d, %d) :: (%d, %d)\n", new_w, new_h, w, h, W, H);
	gl::current_width = w; gl::current_height = h;
	if (w != new_w || h != new_h) // another round
		glutReshapeWindow(w, h);
	else
		ui::nvui.reshape(w, h);
}

void idle() { glutPostRedisplay(); }

// kbd, 4 callbacks, no less.
template<bool down> void keyboard_regular(unsigned char key, int x, int y) {
	if (down) ui::submit_key<false>(key, x, y);
	if (ui::nvui_wants_kbd) return;
	int mod = glutGetModifiers();
	ui::details::kdb_event_t e;
	e.bits = 0;
	e.key        = key;
	e.is_special = false;
	e.is_down    = down;
	e.mod_shift  = mod & GLUT_ACTIVE_SHIFT ? 1 : 0;
	e.mod_ctrl   = mod & GLUT_ACTIVE_CTRL ? 1 : 0;
	e.mod_alt    = mod & GLUT_ACTIVE_ALT ? 1 : 0;
	ui::kbd(e, point_t(x, y));
}
template<bool down> void keyboard_special(int key, int x, int y) {
	if (down) ui::submit_key<true>(key, x, y);
	if (ui::nvui_wants_kbd) return;
	int mod = glutGetModifiers();
	ui::details::kdb_event_t e;
	e.bits = 0;
	e.key        = key;
	e.is_special = true;
	e.is_down    = down;
	e.mod_shift  = mod & GLUT_ACTIVE_SHIFT ? 1 : 0;
	e.mod_ctrl   = mod & GLUT_ACTIVE_CTRL ? 1 : 0;
	e.mod_alt    = mod & GLUT_ACTIVE_ALT ? 1 : 0;
	ui::kbd(e, point_t(x, y));
}

// we'll pull mouse events from nvui.
void mouse(int button, int state, int x, int y) {
	ui::nvui.mouse(button, state, x,y);
}
void motion(int x, int y) {
	ui::nvui.mouseMotion(x,y);
}

void menu(int i) {
	printf("menu: %d\n", i);
}



static int usage(const char * const s) {
	printf("%s [-w width] [-h height] [\"scene filename\"]\n", s);
	return 0;
}

int main(int argc, char *argv[]) {
	const char title[] = "the once known as smallpt.";
	//FIXME: both CUDA and GLUT want to parse that command line, find a way to gracefuly integrate it all.
	int w = window_width, h = window_height;
	const char *filename = "default.scene";
	for (int i=1; i<argc; ++i)
		if (argv[i][0] == '-')
			switch(argv[i][1]) {
				case 'w': if (++i < argc) w = std::atoi(argv[i]); break;
				case 'h': if (++i < argc) h = std::atoi(argv[i]); break;
				default: // assume it's some argument for cuda/glut, skip
					++i;
			}
		else
			filename = argv[i];

	{	// CUDA setup.
		CUT_DEVICE_INIT(argc, argv);
		int dev;
		cudaGetDevice(&dev);
		cudaDeviceProp prop;
		misc::wipe(prop);
		cudaGetDeviceProperties(&prop, dev);
		if (prop.major == 9999 && prop.minor == 9999)
			fatal("no CUDA support?!");

		misc::wipe(params);
		params.framebuffer = 0;
		params.gamma = 2.2;
		params.is_progressive = false;
		params.max_paths = 4;
		params.num_spheres = 0;
		params.pass = 0;
		params.regs_per_block = prop.regsPerBlock;
		params.spp = SS*SS;
		params.verbose = true;
	}
	{	// OpenGL / GLUT setup
		glutInit(&argc, argv);
		glutInitDisplayString(glut_init_display_string);
		glutInitWindowSize(w, h);
		glutCreateWindow(title);
		{
			int doublebuffer = glutGet(GLUT_WINDOW_DOUBLEBUFFER);
			int depth = glutGet(GLUT_WINDOW_DEPTH_SIZE);
			int multi = glutGet(GLUT_WINDOW_NUM_SAMPLES);
			printf("glut: doublebuffer %s depth bits %d multisamples %d\n", doublebuffer ? "yes" : "no", depth, multi);

			int ms_buf, ms;
			glGetIntegerv (GL_SAMPLE_BUFFERS_ARB, &ms_buf);
			glGetIntegerv (GL_SAMPLES_ARB, &ms);
			printf("OpenGl: %d sample buffers, %d samples.\n", ms_buf, ms);
		}

		// initialize necessary OpenGL extensions
		glewInit();
		if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object GL_EXT_framebuffer_object"))
		fatal("missing OpenGL 2.0+PBO+FBO");

		ui::init();
		gl::init(point_t(w, h), filename);

		// register callbacks
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard_regular<1>);
		glutKeyboardUpFunc(keyboard_regular<0>);
		glutSpecialFunc(keyboard_special<1>);
		glutSpecialUpFunc(keyboard_special<0>);
		glutReshapeFunc(reshape);
		glutIdleFunc(idle);

		glutMouseFunc(mouse);
		glutMotionFunc(motion);

		glutIgnoreKeyRepeat(true);
	}

	glutMainLoop();
	return -1;
}
