//
// Created by derue on 15/12/15.
//

#ifndef SLIC_CUDA_FUNUTILS_H
#define SLIC_CUDA_FUNUTILS_H

#endif //SLIC_CUDA_FUNUTILS_H


/* determine width and height (integer value) of an initial spx from its expected size d. */
void getWlHl(int w, int h, int d, int & wl, int & hl);
inline int iDivUp(int a, int b){ return (a%b==0)? a/b : a/b+1; }