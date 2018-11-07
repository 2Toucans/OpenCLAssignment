__kernel void myKernel(__global int* inData, __global int* outData, int width, int height)
{
	union {
		unsigned int rgba;
		struct {
			char r;
			char g;
			char b;
			char a;
		};
	} colour;

	outData[get_global_id(0)] = inData[get_global_id(0)];
}