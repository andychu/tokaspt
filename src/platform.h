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
#ifndef PLATFORM_H
#define PLATFORM_H

#include "specifics.h"
#ifdef WIN32
	//note: this header is supposed to be independent from previous includes, but it's not
	//      because that would require pulling in those horrible windows headers.
	namespace win32 {
		enum { magic_fs_path_len = 260 }; // so says Microsoft.
		// invokes either a load or save file selection dialog.
		template<bool is_load, size_t buf_size>
			static bool_t dialog_load_save(const char * const title, char (&buf)[buf_size]) {
				OPENFILENAME crap = { 0 };
				buf[0] = 0;
				if (!is_load && buf_size > 14) sprintf(buf, "blah.00.scene");
				crap.lStructSize = sizeof(crap);
				crap.lpstrFile = buf;
				crap.nMaxFile = buf_size;
				crap.lpstrFilter = "All\0*.*\0scene description\0*.scene\0";
				crap.nFilterIndex = 2;
				crap.lpstrTitle = title;
				crap.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | is_load ? OFN_FILEMUSTEXIST : OFN_OVERWRITEPROMPT;
				return is_load ? GetOpenFileName(&crap) : GetSaveFileName(&crap);
			}
	}
	namespace platform = win32;
#elif defined linux
	namespace LiNuX {
		enum { magic_fs_path_len = 512 };
		// no, i'm not ashamed. it's short an simple, right? heh.
		template<bool is_load, size_t buf_size>
			static bool_t dialog_load_save(const char * const title, char (&buf)[buf_size]) {
				sys::fmt_t cmd("zenity --file-selection %s --title=\"%s\"", is_load ? "" : "--save", title);
				if (FILE *f = popen(cmd, "r")) {
					unsigned i = 0;
					do {
						int c = fgetc(f);
						if (c == EOF || c == 0 || c == 0xA) break;
						buf[i] = c;
					} while (++i < buf_size);
					buf[i] = 0;
					fclose(f);
					return i > 0;
				}
				return false;
			}
	}
	namespace platform = LiNuX;
#else
	#error "platform specific code sorely missing."
#endif

#endif
