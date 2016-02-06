//
// Created by derue on 15/12/15.
//

#include "funUtilsSC.h"



void getWlHl(int w, int h, int d, int& wl, int& hl) {

    int wl1, wl2;
    int hl1, hl2;
    wl1 = wl2 = d;
    hl1 = hl2 = d;

    while ((w%wl1)!=0) {
        wl1++;
    }

    while ((w%wl2)!= 0) {
        wl2--;
    }
    while ((h%hl1) != 0) {
        hl1++;
    }

    while ((h%hl2) != 0) {
        hl2--;
    }
    wl = ((d - wl2) < (wl1 - d)) ? wl2 : wl1;
    hl = ((d - hl2) < (hl1 - d)) ? hl2 : hl1;
}