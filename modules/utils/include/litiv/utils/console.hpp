
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "litiv/utils/defines.hpp"
#include "litiv/utils/cxx.hpp"

/**
 *        console.hpp (litiv version inspired from rlutil.h)
 *
 *            rlutil.h: Copyright (C) 2010 Tapio Vierros
 *  -- see https://github.com/tapio/rlutil for more information --
 *      (originally distributed under the WTF public license)
 */

// define this to use ANSI escape sequences also on Windows (defaults to using WinAPI otherwise)
#if 0
#define RLUTIL_USE_ANSI
#endif
// define/typedef this to your preference to override rlutil's string type
#if 0
#define RLUTIL_STRING_T char*
#endif

#include <iostream>
#include <string>
#include <sstream>

#ifdef _WIN32
#include <windows.h>  // for WinAPI and Sleep()
#define _NO_OLDNAMES  // for MinGW compatibility
#include <conio.h>    // for getch() and kbhit()
#define getch _getch
#define kbhit _kbhit
#else //ndef _WIN32
#include <cstdio> // for getch()
#include <termios.h> // for getch() and kbhit()
#include <unistd.h> // for getch(), kbhit() and (u)sleep()
#include <sys/ioctl.h> // for getkey()
#include <sys/types.h> // for kbhit()
#include <sys/time.h> // for kbhit()

// get character without waiting for return to be pressed; windows has this in conio.h
inline int getch(void) {
    // Here be magic.
    struct termios oldt,newt;
    int ch;
    tcgetattr(STDIN_FILENO,&oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO,TCSANOW,&newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO,TCSANOW,&oldt);
    return ch;
}

// determines if keyboard has been hit; windows has this in conio.h
inline int kbhit(void) {
    // Here be dragons.
    static struct termios oldt,newt;
    int cnt = 0;
    tcgetattr(STDIN_FILENO,&oldt);
    newt = oldt;
    newt.c_lflag    &= ~(ICANON | ECHO);
    newt.c_iflag     = 0; // input mode
    newt.c_oflag     = 0; // output mode
    newt.c_cc[VMIN]  = 1; // minimum time to wait
    newt.c_cc[VTIME] = 1; // minimum characters to wait for
    tcsetattr(STDIN_FILENO,TCSANOW,&newt);
    ioctl(0,FIONREAD,&cnt); // Read count
    struct timeval tv;
    tv.tv_sec  = 0;
    tv.tv_usec = 100;
    select(STDIN_FILENO+1,NULL,NULL,NULL,&tv); // A small time delay
    tcsetattr(STDIN_FILENO,TCSANOW,&oldt);
    return cnt; // Return number of characters
}

#endif //ndef _WIN32

namespace rlutil {

#ifndef RLUTIL_STRING_T
    typedef std::string RLUTIL_STRING_T;
#endif //ndef RLUTIL_STRING_T
    inline void RLUTIL_PRINT(RLUTIL_STRING_T st) { std::cout << st; }

    enum ColorCode {
        Color_BLACK=0,
        Color_BLUE,
        Color_GREEN,
        Color_CYAN,
        Color_RED,
        Color_MAGENTA,
        Color_BROWN,
        Color_GREY,
        Color_DARKGREY,
        Color_LIGHTBLUE,
        Color_LIGHTGREEN,
        Color_LIGHTCYAN,
        Color_LIGHTRED,
        Color_LIGHTMAGENTA,
        Color_BOLDYELLOW,
        Color_BOLDWHITE,
        Color_RESET,
        // codes below are not supported on windows
        Color_BOLDGREEN,
        Color_BOLDRED,
    };

    /**
     * ANSI color strings
     *
     * ANSI_CLS - Clears screen
     * ANSI_BLACK - Black
     * ANSI_RED - Red
     * ANSI_GREEN - Green
     * ANSI_BROWN - Brown / dark yellow
     * ANSI_BLUE - Blue
     * ANSI_MAGENTA - Magenta / purple
     * ANSI_CYAN - Cyan
     * ANSI_GREY - Grey / dark white
     * ANSI_DARKGREY - Dark grey / light black
     * ANSI_LIGHTRED - Light red
     * ANSI_LIGHTGREEN - Light green
     * ANSI_BOLDYELLOW - Yellow (bold/bright)
     * ANSI_LIGHTBLUE - Light blue
     * ANSI_LIGHTMAGENTA - Light magenta / light purple
     * ANSI_LIGHTCYAN - Light cyan
     * ANSI_BOLDWHITE - White (bold/bright)
     * ANSI_RESET - Resets formatting
     */
    const RLUTIL_STRING_T ANSI_CLS = "\033[2J";
    const RLUTIL_STRING_T ANSI_BLACK = "\033[22;30m";
    const RLUTIL_STRING_T ANSI_RED = "\033[22;31m";
    const RLUTIL_STRING_T ANSI_BOLDRED = "\033[22;31;1m";
    const RLUTIL_STRING_T ANSI_GREEN = "\033[22;32m";
    const RLUTIL_STRING_T ANSI_BOLDGREEN = "\033[22;32;1m";
    const RLUTIL_STRING_T ANSI_BROWN = "\033[22;33m";
    const RLUTIL_STRING_T ANSI_BLUE = "\033[22;34m";
    const RLUTIL_STRING_T ANSI_MAGENTA = "\033[22;35m";
    const RLUTIL_STRING_T ANSI_CYAN = "\033[22;36m";
    const RLUTIL_STRING_T ANSI_GREY = "\033[22;37m";
    const RLUTIL_STRING_T ANSI_DARKGREY = "\033[01;30m";
    const RLUTIL_STRING_T ANSI_LIGHTRED = "\033[01;31m";
    const RLUTIL_STRING_T ANSI_LIGHTGREEN = "\033[01;32m";
    const RLUTIL_STRING_T ANSI_BOLDYELLOW = "\033[01;33m";
    const RLUTIL_STRING_T ANSI_LIGHTBLUE = "\033[01;34m";
    const RLUTIL_STRING_T ANSI_LIGHTMAGENTA = "\033[01;35m";
    const RLUTIL_STRING_T ANSI_LIGHTCYAN = "\033[01;36m";
    const RLUTIL_STRING_T ANSI_BOLDWHITE = "\033[01;37m";
    const RLUTIL_STRING_T ANSI_RESET = "\033[39;49m\033[0m";

    /**
     * Key codes for keyhit()
     *
     * KEY_ESCAPE  - Escape
     * KEY_ENTER   - Enter
     * KEY_SPACE   - Space
     * KEY_INSERT  - Insert
     * KEY_HOME    - Home
     * KEY_END     - End
     * KEY_DELETE  - Delete
     * KEY_PGUP    - PageUp
     * KEY_PGDOWN  - PageDown
     * KEY_UP      - Up arrow
     * KEY_DOWN    - Down arrow
     * KEY_LEFT    - Left arrow
     * KEY_RIGHT   - Right arrow
     * KEY_F1      - F1
     * KEY_F2      - F2
     * KEY_F3      - F3
     * KEY_F4      - F4
     * KEY_F5      - F5
     * KEY_F6      - F6
     * KEY_F7      - F7
     * KEY_F8      - F8
     * KEY_F9      - F9
     * KEY_F10     - F10
     * KEY_F11     - F11
     * KEY_F12     - F12
     * KEY_NUMDEL  - Numpad del
     * KEY_NUMPAD0 - Numpad 0
     * KEY_NUMPAD1 - Numpad 1
     * KEY_NUMPAD2 - Numpad 2
     * KEY_NUMPAD3 - Numpad 3
     * KEY_NUMPAD4 - Numpad 4
     * KEY_NUMPAD5 - Numpad 5
     * KEY_NUMPAD6 - Numpad 6
     * KEY_NUMPAD7 - Numpad 7
     * KEY_NUMPAD8 - Numpad 8
     * KEY_NUMPAD9 - Numpad 9
     */
    const int KEY_ESCAPE  = 0;
    const int KEY_ENTER   = 1;
    const int KEY_SPACE   = 32;

    const int KEY_INSERT  = 2;
    const int KEY_HOME    = 3;
    const int KEY_PGUP    = 4;
    const int KEY_DELETE  = 5;
    const int KEY_END     = 6;
    const int KEY_PGDOWN  = 7;

    const int KEY_UP      = 14;
    const int KEY_DOWN    = 15;
    const int KEY_LEFT    = 16;
    const int KEY_RIGHT   = 17;

    const int KEY_F1      = 18;
    const int KEY_F2      = 19;
    const int KEY_F3      = 20;
    const int KEY_F4      = 21;
    const int KEY_F5      = 22;
    const int KEY_F6      = 23;
    const int KEY_F7      = 24;
    const int KEY_F8      = 25;
    const int KEY_F9      = 26;
    const int KEY_F10     = 27;
    const int KEY_F11     = 28;
    const int KEY_F12     = 29;

    const int KEY_NUMDEL  = 30;
    const int KEY_NUMPAD0 = 31;
    const int KEY_NUMPAD1 = 127;
    const int KEY_NUMPAD2 = 128;
    const int KEY_NUMPAD3 = 129;
    const int KEY_NUMPAD4 = 130;
    const int KEY_NUMPAD5 = 131;
    const int KEY_NUMPAD6 = 132;
    const int KEY_NUMPAD7 = 133;
    const int KEY_NUMPAD8 = 134;
    const int KEY_NUMPAD9 = 135;

    // reads a key press (blocking) and returns a key code; see key codes for info (note: only Arrows, Esc, Enter and Space are currently working properly)
    inline int getkey(void) {
#ifndef _WIN32
        int cnt = kbhit(); // for ANSI escapes processing
#endif //ndef _WIN32
        int k = getch();
        switch(k) {
            case 0: {
                int kk;
                switch (kk = getch()) {
                    case 71: return KEY_NUMPAD7;
                    case 72: return KEY_NUMPAD8;
                    case 73: return KEY_NUMPAD9;
                    case 75: return KEY_NUMPAD4;
                    case 77: return KEY_NUMPAD6;
                    case 79: return KEY_NUMPAD1;
                    case 80: return KEY_NUMPAD4;
                    case 81: return KEY_NUMPAD3;
                    case 82: return KEY_NUMPAD0;
                    case 83: return KEY_NUMDEL;
                    default: return kk-59+KEY_F1; // Function keys
                }}
            case 224: {
                int kk;
                switch (kk = getch()) {
                    case 71: return KEY_HOME;
                    case 72: return KEY_UP;
                    case 73: return KEY_PGUP;
                    case 75: return KEY_LEFT;
                    case 77: return KEY_RIGHT;
                    case 79: return KEY_END;
                    case 80: return KEY_DOWN;
                    case 81: return KEY_PGDOWN;
                    case 82: return KEY_INSERT;
                    case 83: return KEY_DELETE;
                    default: return kk-123+KEY_F1; // Function keys
                }}
            case 13: return KEY_ENTER;
#ifdef _WIN32
            case 27: return KEY_ESCAPE;
#else //ndef _WIN32
            case 155: // single-character CSI
            case 27: {
                // Process ANSI escape sequences
                if(cnt >= 3 && getch() == '[') {
                    switch (k = getch()) {
                        case 'A': return KEY_UP;
                        case 'B': return KEY_DOWN;
                        case 'C': return KEY_RIGHT;
                        case 'D': return KEY_LEFT;
                    }
                } else return KEY_ESCAPE;
            }
#endif //ndef _WIN32
            default: return k;
        }
    }

    // non-blocking getch(); returns 0 if no key was pressed
    inline int nb_getch(void) {
        if(kbhit())
            return getch();
        else
            return 0;
    }

    // returns ANSI color escape sequence for specified color enum
    inline RLUTIL_STRING_T getANSIColor(const ColorCode c) {
        switch(c) {
            case Color_BLACK : return ANSI_BLACK;
            case Color_BLUE : return ANSI_BLUE; // non-ANSI
            case Color_GREEN : return ANSI_GREEN;
            case Color_BOLDGREEN : return ANSI_BOLDGREEN;
            case Color_CYAN : return ANSI_CYAN; // non-ANSI
            case Color_RED : return ANSI_RED; // non-ANSI
            case Color_BOLDRED : return ANSI_BOLDRED;
            case Color_MAGENTA : return ANSI_MAGENTA;
            case Color_BROWN : return ANSI_BROWN;
            case Color_GREY : return ANSI_GREY;
            case Color_DARKGREY : return ANSI_DARKGREY;
            case Color_LIGHTBLUE : return ANSI_LIGHTBLUE; // non-ANSI
            case Color_LIGHTGREEN: return ANSI_LIGHTGREEN;
            case Color_LIGHTCYAN: return ANSI_LIGHTCYAN; // non-ANSI;
            case Color_LIGHTRED: return ANSI_LIGHTRED; // non-ANSI;
            case Color_LIGHTMAGENTA: return ANSI_LIGHTMAGENTA;
            case Color_BOLDYELLOW: return ANSI_BOLDYELLOW; // non-ANSI
            case Color_BOLDWHITE: return ANSI_BOLDWHITE;
            case Color_RESET: return ANSI_RESET;
            default: return "";
        }
    }

    // changes color specified by enum (Windows / QBasic colors)
    inline void setColor(ColorCode c) {
#if defined(_WIN32) && !defined(RLUTIL_USE_ANSI)
        if(c==Color_RESET)
            system("color 07");
        else if(c<Color_RESET) {
            HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
            SetConsoleTextAttribute(hConsole,(WORD)c);
        }
#else //(!defined(_WIN32) || defined(RLUTIL_USE_ANSI))
        RLUTIL_PRINT(getANSIColor(c));
#endif //(!defined(_WIN32) || defined(RLUTIL_USE_ANSI))
    }

    // clears screen and moves cursor home
    inline void cls() {
#if defined(_WIN32) && !defined(RLUTIL_USE_ANSI)
        system("cls");
#else //(!defined(_WIN32) || defined(RLUTIL_USE_ANSI))
        RLUTIL_PRINT("\033[2J\033[H");
#endif
    }

    // sets the cursor position to 1-based x,y
    inline void locate(int x, int y) {
#if defined(_WIN32) && !defined(RLUTIL_USE_ANSI)
        COORD coord;
        coord.X = (SHORT)x-1;
        coord.Y = (SHORT)y-1; // Windows uses 0-based coordinates
        SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE),coord);
#else //(!defined(_WIN32) || defined(RLUTIL_USE_ANSI))
        std::ostringstream oss;
        oss << "\033[" << y << ";" << x << "H";
        RLUTIL_PRINT(oss.str());
#endif //(!defined(_WIN32) || defined(RLUTIL_USE_ANSI))
    }

    // hides the cursor
    inline void hidecursor() {
#if defined(_WIN32) && !defined(RLUTIL_USE_ANSI)
        HANDLE hConsoleOutput;
        CONSOLE_CURSOR_INFO structCursorInfo;
        hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        GetConsoleCursorInfo(hConsoleOutput,&structCursorInfo); // Get current cursor size
        structCursorInfo.bVisible = FALSE;
        SetConsoleCursorInfo(hConsoleOutput,&structCursorInfo);
#else //(!defined(_WIN32) || defined(RLUTIL_USE_ANSI))
        RLUTIL_PRINT("\033[?25l");
#endif //(!defined(_WIN32) || defined(RLUTIL_USE_ANSI))
    }

    // shows the cursor
    inline void showcursor() {
#if defined(_WIN32) && !defined(RLUTIL_USE_ANSI)
        HANDLE hConsoleOutput;
        CONSOLE_CURSOR_INFO structCursorInfo;
        hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        GetConsoleCursorInfo(hConsoleOutput,&structCursorInfo); // Get current cursor size
        structCursorInfo.bVisible = TRUE;
        SetConsoleCursorInfo(hConsoleOutput,&structCursorInfo);
#else //(!defined(_WIN32) || defined(RLUTIL_USE_ANSI))
        RLUTIL_PRINT("\033[?25h");
#endif //(!defined(_WIN32) || defined(RLUTIL_USE_ANSI))
    }

    // waits given number of milliseconds before continuing
    inline void msleep(unsigned int ms) {
#ifdef _WIN32
        Sleep(ms);
#else //ndef win32
        // usleep argument must be under 1 000 000
        if(ms > 1000) sleep(ms/1000000);
        usleep((ms % 1000000) * 1000);
#endif //ndef win32
    }

    // get the number of rows in the terminal window, or -1 on error
    inline int trows() {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if(!GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),&csbi))
            return -1;
        else
            return csbi.srWindow.Bottom - csbi.srWindow.Top + 1; // Window height
            // return csbi.dwSize.Y; // Buffer height
#else //ndef win32
#ifdef TIOCGSIZE
        struct ttysize ts;
        if(ioctl(STDIN_FILENO,TIOCGSIZE,&ts))
            return -1;
        return ts.ts_lines;
#elif defined(TIOCGWINSZ)
        struct winsize ts;
        if(ioctl(STDIN_FILENO,TIOCGWINSZ,&ts))
            return -1;
        return ts.ws_row;
#else //ndef TIOCGSIZE
        return -1;
#endif //ndef TIOCGSIZE
#endif //ndef _WIN32
    }

    // get the number of columns in the terminal window or -1 on error
    inline int tcols() {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if(!GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),&csbi))
            return -1;
        else
            return csbi.srWindow.Right - csbi.srWindow.Left + 1; // Window width
            // return csbi.dwSize.X; // Buffer width
#else //ndef _WIN32
#ifdef TIOCGSIZE
        struct ttysize ts;
        if(ioctl(STDIN_FILENO,TIOCGSIZE,&ts))
            return -1;
        return ts.ts_cols;
#elif defined(TIOCGWINSZ)
        struct winsize ts;
        if(ioctl(STDIN_FILENO,TIOCGWINSZ,&ts))
            return -1;
        return ts.ws_col;
#else //ndef TIOCGSIZE
        return -1;
#endif //ndef TIOCGSIZE
#endif //ndef _WIN32
    }

    // waits until a key is pressed
    inline void anykey() {
        // TODO: Allow optional message for anykey()?
        getch();
    }

} // namespace rlutil

namespace lv {

#if defined(_MSC_VER)
    /// sets the console window to a certain size (with optional buffer resizing)
    void SetConsoleWindowSize(int x, int y, int buffer_lines=-1) {
        // derived from http://www.cplusplus.com/forum/windows/121444/
        HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
        if(h==INVALID_HANDLE_VALUE)
            lvError("SetConsoleWindowSize: Unable to get stdout handle");
        COORD largestSize = GetLargestConsoleWindowSize(h);
        if(x>largestSize.X)
            x = largestSize.X;
        if(y>largestSize.Y)
            y = largestSize.Y;
        if(buffer_lines<=0)
            buffer_lines = y;
        CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
        if(!GetConsoleScreenBufferInfo(h,&bufferInfo))
            lvError("SetConsoleWindowSize: Unable to retrieve screen buffer info");
        SMALL_RECT& winInfo = bufferInfo.srWindow;
        COORD windowSize = {winInfo.Right-winInfo.Left+1,winInfo.Bottom-winInfo.Top+1};
        if(windowSize.X>x || windowSize.Y>y) {
            SMALL_RECT info = {0,0,SHORT((x<windowSize.X)?(x-1):(windowSize.X-1)),SHORT((y<windowSize.Y)?(y-1):(windowSize.Y-1))};
            if(!SetConsoleWindowInfo(h,TRUE,&info))
                lvError("SetConsoleWindowSize: Unable to resize window before resizing buffer");
        }
        COORD size = {SHORT(x),SHORT(y)};
        if(!SetConsoleScreenBufferSize(h,size))
            lvError("SetConsoleWindowSize: Unable to resize screen buffer");
        SMALL_RECT info = {0,0,SHORT(x-1),SHORT(y-1)};
        if(!SetConsoleWindowInfo(h, TRUE, &info))
            lvError("SetConsoleWindowSize: Unable to resize window after resizing buffer");
    }
#endif //defined(_MSC_VER)

    /// shows a progression bar in the console
    void updateConsoleProgressBar(const std::string& sMsg, float fCompletion, size_t nBarCols=20) {
        if(nBarCols==0)
            return;
        const int nRows = rlutil::trows();
        const int nCols = rlutil::tcols();
        printf("\r%s  ",sMsg.c_str());
        if(nRows>0 && nCols>0)
            rlutil::setColor(rlutil::Color_BOLDWHITE);
        printf("[");
        const size_t nComplBars = size_t(fCompletion*nBarCols);
        for(size_t n=0; n<nBarCols; ++n) {
            if(nRows>0 && nCols>0) {
                if(n<nBarCols/3)
                    rlutil::setColor(rlutil::Color_BOLDRED);
                else if(n>=(nBarCols-nBarCols/3))
                    rlutil::setColor(rlutil::Color_BOLDGREEN);
                else
                    rlutil::setColor(rlutil::Color_BOLDYELLOW);
            }
            if(n<=nComplBars)
                printf("=");
            else
                printf(" ");
        }
        if(nRows>0 && nCols>0)
            rlutil::setColor(rlutil::Color_BOLDWHITE);
        printf("]");
        if(nRows>0 && nCols>0)
            rlutil::setColor(rlutil::Color_RESET);
        printf(" ");
        fflush(stdout);
    }

    /// cleans a specific row from the console (default=last)
    void cleanConsoleRow(int nRowIdx=INT_MAX) {
        if(nRowIdx<0)
            return;
        const int nRows = rlutil::trows();
        const int nCols = rlutil::tcols();
        if(nRows>0 && nCols>0)
            printf("\r%s\r",std::string(nCols,' ').c_str());
        else
            printf("\r");
        fflush(stdout);
    }

} // namespace lv
