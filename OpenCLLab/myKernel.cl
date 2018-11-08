union Colour {
	unsigned int rgba;
	struct {
		unsigned char r;
		unsigned char g;
		unsigned char b;
		unsigned char a;
	};
};

// 5x5 Gaussian Kernel
__constant float gauss[25] = {
	0.058379266789788185f,0.06705176688958259f,0.058379266789788185f,0.038530430847356856f,0.01927728703018819f,
	0.06705176688958259f,0.0770126055060589f,0.06705176688958259f,0.044254297962235976f,0.022141013878541074f,
	0.058379266789788185f,0.06705176688958259f,0.058379266789788185f,0.038530430847356856f,0.01927728703018819f,
	0.038530430847356856f,0.044254297962235976f,0.038530430847356856f,0.025430160105104932f,0.012723047336580126f,
	0.01927728703018819f,0.022141013878541074f,0.01927728703018819f,0.012723047336580126f,0.006365509806458638f
};

__constant int gaussSize = 5;

// 3x3 Vertical Edge Detection Kernel
__constant float vedge[9] = {
	+1,	+2,	+1,
	0,	0,	0,
	-1, -2, -1
};

__constant int vedgeSize = 3;

// 3x3 Horizontal Edge Detection Kernel
__constant float hedge[9] = {
	+1, 0, -1,
	+2, 0, -2,
	+1, 0, -1
};

__constant int hedgeSize = 3;

inline union Colour greyscale(union Colour c) {
	char avg = (c.r + c.g + c.b) / 3;
	c.r = avg;
	c.g = avg;
	c.b = avg;
	return c;
}

inline union Colour getPixel(__global int* inData, int x, int y, int w, int h) {
	x = clamp(x, 0, w - 1);
	y = clamp(y, 0, h - 1);
	union Colour c = { inData[y * w + x] };
	return c;
}

inline union Colour convolve(__global int* inData, int x, int y, int w, int h, int type) {
	int kernelSize = 5;
	union Colour colour = { inData[get_global_id(0)] };
	int mid = kernelSize / 2;
	float accR = 0;
	float accG = 0;
	float accB = 0;
	for (int i = 0; i < kernelSize; i++) {
		for (int j = 0; j < kernelSize; j++) {
			int kernelPos = i * kernelSize + j;
			union Colour col = getPixel(inData, x + j - mid, y + i - mid, w, h);
			accR += convert_float(col.r) * gauss[kernelPos];
			accG += convert_float(col.g) * gauss[kernelPos];
			accB += convert_float(col.b) * gauss[kernelPos];
		}
	}
	colour.r = convert_uchar(accR);
	colour.g = convert_uchar(accG);
	colour.b = convert_uchar(accB);
	return colour;
}

__kernel void myKernel(__global int* inData, __global int* outData, int width, int height)
{
	int x = get_global_id(0) % width;
	int y = get_global_id(0) / width;
	//union Colour colour = { inData[get_global_id(0)] };
	//colour = greyscale(colour);

	union Colour colour = convolve(inData, x, y, width, height, 0);
	union Colour c2 = { 0 };
	c2.r = (x * 255) / width;
	c2.g = (y * 255) / height;
	c2.a = 255;

	outData[get_global_id(0)] = colour.rgba;
}