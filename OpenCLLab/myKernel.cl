union Colour
{
	unsigned int rgba;
	struct
	{
		char r;
		char g;
		char b;
		char a;
	};
};

__kernel void myKernel(__global int* inData, __global int* outData, int width, int height)
{
	union Colour colour = {inData[get_global_id(0)]};

	colour.g = 0;
	colour.b = 0;

	outData[get_global_id(0)] = colour.rgba;
}