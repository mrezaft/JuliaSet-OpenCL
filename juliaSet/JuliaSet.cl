typedef struct _Julia_region {
	
	float2		pos; // Position of region (in normalised image coordinates [0, 1]) - float4 alignment
	float3		colour; // Colour of region - float4 alignment
} Julia_region;

kernel void Julia(
const global Julia_region *R,
 write_only image2d_t outputImage,
  const int mOfIteration)
{
	// Get id of element in array
	int x = get_global_id(0);
	int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);
	
	// Calculate normalised coordinates of current pixel in [0, 1] range
	
	
	float4 P;

//	P.x = (float)(x) / (float)(w-1);
//	P.y = (float)(y) / (float)(h-1);


	//P.x = 1.1*(float)(w / 2 - x) / (w / 2);
	//P.y =  1.1*(float)(h / 2 - y) / (h / 2);

	P.x = 4*(float)(w / 2 - x) / (w );
	P.y =  4*(float)(h / 2 - y) / (h );
	P.z = 0.0f;
	P.w = 0.0f;
	

	

	int i = 0;

	float2 c;
	//c.x = -0.805;
	//c.y = 0.156;

	//c.x= 0.70176;
	//c.y=0.3842;
	
	//c.x=-0.835;
	//c.y=0.2321;

	c.x=-0.835 * 1.5;
	c.y=0.2321 *0.5;

	//c.x=-0.8;
	//c.y=0.156;
	//c.x=  0.065;
	//c.y= 0.122;

	float2 z;
	z.x = P.x;
	z.y = P.y;

	float2 z1;

	for (i = 0; i < mOfIteration && (length(z) < 2.0); i++)
	//for (i = 0; i < mOfIteration && (sqrt(z.x*z.x+z.y*z.y) < 2.0); i++)
	{
		z1.x = z.x * z.x - z.y * z.y;
		z1.y = 2 * z.x * z.y;
		
		
		//z.x = z1.x + c.x;
		//z.y = z1.y + c.y;

	//	z.x = z1.x*z1.x*z1.x*z1.x  + c.x;
	//	z.y = z1.y*z1.y*z1.y*z1.y + c.y;



		z.x = sqrt(sinh(z1.x*z1.x)) + c.x;
		z.y = sqrt(sinh(z1.y*z1.y)) + c.y;

	}

	float4 C;

	C.x = R[i].colour.x;
	C.y = R[i].colour.y;
	C.z = R[i].colour.z;
	C.w = 1.0;

	write_imagef(outputImage, (int2)(x, y), C);
}



