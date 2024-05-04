#include <iostream>
#include <fstream>
#include <vector>


int main() {

	//	generating masks

	int masks_i64[10][784] = { 0 };
	int masks_max[10] = { 0 };
	float masks_f32[10][784] = { 0.0 };

	std::ifstream input_data("train-images-idx3-ubyte", std::ios::binary);
	std::vector<unsigned char> data(std::istreambuf_iterator<char>(input_data), {});
	input_data.close();

	std::ifstream input_labels("train-labels-idx1-ubyte", std::ios::binary);
	std::vector<unsigned char> labels(std::istreambuf_iterator<char>(input_data), {});
	input_data.close();

	for (int i = 8; i < 60008; i++) {		// offset by 8 for labels, by 16 for images
		for (int p = 0; p < 784; p++) {
			masks_i64[labels[i]][p] += data[8 + p + i * 784];
			if (masks_max[labels[i]] < masks_i64[labels[i]][p]) {
				masks_max[labels[i]] = masks_i64[labels[i]][p];
			}
		}
	}

	for (int i = 0; i < 10; i++) {
		for (int p = 0; p < 784; p++) {
			masks_f32[i][p] = (float)masks_i64[i][p] / (float)masks_max[i];
		}
	}

	std::ofstream myfile;
	myfile.open("masks.txt");
	for (int d = 0; d < 10; d++) {
		for (int p = 0; p < 784; p++) {
			myfile << masks_f32[d][p] << "; ";
		}
		myfile << "\n";
	}
	myfile.close();

	return 0;
}