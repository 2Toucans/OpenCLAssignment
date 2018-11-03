__kernel void myKernel(__global int* inData, __global int* outData)
{
	//outData[get_global_id(0)] = inData[get_global_id(0)] * 2;

	size_t id = get_global_id(1) * get_global_size(0) + get_global_id(0);

	outData[id] = inData[id] * 2;
}