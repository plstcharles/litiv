/*//////////////////////////////////////////////////////////////////////////////////////////////////
///  Image.h   A minimal PGM image class
///  Version 1.01
////////////////////////////////////////////////////////////////////////////////////////////////////

Copyright 2009 Hiroshi Ishikawa.
This software can be used for research purposes only.
This software or its derivatives must not be publicly distributed
without a prior consent from the author (Hiroshi Ishikawa).

THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For the latest version, check: http://www.nsc.nagoya-cu.ac.jp/~hi/

////////////////////////////////////////////////////////////////////////////////////////////////////

A minimal PGM image class for use in HOCRdemo.cpp.

//////////////////////////////////////////////////////////////////////////////////////////////////*/


#if !defined(IMAGE_H)
#define IMAGE_H

#include <vector>
#include <fstream>
#include <math.h>


struct image
{
	image() : W(0), H(0) {}
	image(int w, int h) : W(w), H(h) {buf.resize(W * H);}
	~image() {}
	bool empty() const {return buf.empty();}
	const unsigned char& operator()(int x, int y) const {return buf[x + y * W];}
	bool readPGM(std::string fn)
	{
		std::ifstream s(fn.c_str());
		if(!s)
			return false;
		char a;
		s >> a;
		if (a != 'P')
			return false;
		s >> a;
		if (a != '2')
			return false;
		char b[5000];
		s.getline(b, 4999);
		s.getline(b, 4999);
		int X;
		s >> W >> H >> X;
		if (X != 255)
			return false;
		buf.resize(W * H);
		std::vector<unsigned char>::iterator p = buf.begin();
		for (int i = 0; i < W * H; i++)
		{
			s >> X;
			*p++ = (unsigned char)X;
		}
		return true;
	}
	void writePGM(std::string fn)
	{
		std::ofstream s(fn.c_str());
		if(s)
		{
			s << "P2\n#output from hocrtest\n" << W << " " << H << "\n" << (int)255 << "\n";
			int c = 0;
			while (c < W * H)
			{
				for (int i = 0; i < 19 && c < W * H; i++)
					s << (int)buf[c++] << " ";
				s << "\n";
			}
		}
	}
	void gaussianblur(image& blurred, double sigma) const
	{
		blurred.buf.resize(W * H);
		int r = int(sigma * 3) + 1;
		double s2 = sigma * sigma * 2;
		for (int y = 0; y < H; y++)
			for (int x = 0; x < W; x++)
			{
				double s = 0;
				for (int v = -r; v <= r; v++)
				{
					int v1 = std::max(0, std::min(H - 1, y + v));
					for (int u = -r; u <= r; u++)
					{
						int u1 = std::max(0, std::min(W - 1, x + u));
						s += exp(-(u * u + v * v) / s2) * buf[v1 * W + u1];
					}
				}
				blurred.buf[y * W + x] = (unsigned char)std::min(255.0, std::max(0.0, (s / s2 / 3.141592653589)));
			}
	}
	int W, H;
	std::vector<unsigned char> buf;
};

#endif
