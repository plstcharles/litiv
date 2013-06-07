#pragma once

static const int s_nSamplesInitPatternWidth = 3;
static const int s_nSamplesInitPatternHeight = 3;
// floor(fspecial('gaussian', 3, 1)*256)
static const int s_nSamplesInitPatternTot = 252;
static const int s_anSamplesInitPattern[9] = {
	19,    31,    19,
	31,    52,    31,
	19,    31,    19,
};


static inline void getRandSamplePosition(int& x_sample, int& y_sample, const int x_orig, const int y_orig, const int border, const cv::Size& imgsize) {
	int r = 1+rand()%s_nSamplesInitPatternTot;
	for(x_sample=0; x_sample<s_nSamplesInitPatternWidth; ++x_sample) {
		for(y_sample=0; y_sample<s_nSamplesInitPatternHeight; ++y_sample) {
			r -= s_anSamplesInitPattern[x_sample*s_nSamplesInitPatternWidth + y_sample];
			if(r<=0)
				goto stop;
		}
	}
	stop:
	x_sample += x_orig-s_nSamplesInitPatternWidth/2;
	y_sample += y_orig-s_nSamplesInitPatternHeight/2;
	if(x_sample<border)
		x_sample = border;
	else if(x_sample>=imgsize.width-border)
		x_sample = imgsize.width-border-1;
	if(y_sample<border)
		y_sample = border;
	else if(y_sample>=imgsize.height-border)
		y_sample = imgsize.height-border-1;
}

// simple 8-connected neighbors
static const int s_anNeighborPatternSize = 8;
static const int s_anNeighborPattern[8][2] = {
	{-1, 1},  { 0, 1},  { 1, 1},
	{-1, 0},            { 1, 0},
	{-1,-1},  { 0,-1},  { 1,-1},
};

static inline void getRandNeighborPosition(int& x_neighbor, int& y_neighbor, const int x_orig, const int y_orig, const int border, const cv::Size& imgsize) {
	int r = rand()%s_anNeighborPatternSize;
	x_neighbor = x_orig+s_anNeighborPattern[r][0];
	y_neighbor = y_orig+s_anNeighborPattern[r][1];
	if(x_neighbor<border)
		x_neighbor = border;
	else if(x_neighbor>=imgsize.width-border)
		x_neighbor = imgsize.width-border-1;
	if(y_neighbor<border)
		y_neighbor = border;
	else if(y_neighbor>=imgsize.height-border)
		y_neighbor = imgsize.height-border-1;
}
