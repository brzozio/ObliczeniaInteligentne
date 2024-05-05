#include <iostream>
#include <fstream>
#include <vector>

int main() {

	constexpr int size_target_header = 8;
	constexpr int size_data_header = 16;
	constexpr int size_target = 10;

	// loading data

	std::ifstream input_train_data("train-images-idx3-ubyte", std::ios::binary);
	if (!input_train_data.is_open()) {
		std::cout << "unable to open source file";
		return -1;
	}
	std::vector<unsigned char> train_data(std::istreambuf_iterator<char>(input_train_data), {});
	input_train_data.close();
	train_data.erase(train_data.begin(), std::next(train_data.begin(), size_data_header));

	std::ifstream input_test_data("t10k-images-idx3-ubyte", std::ios::binary);
	if (!input_test_data.is_open()) {
		std::cout << "unable to open source file";
		return -1;
	}
	std::vector<unsigned char> test_data(std::istreambuf_iterator<char>(input_test_data), {});
	input_test_data.close();
	test_data.erase(test_data.begin(), std::next(test_data.begin(), size_data_header));

	std::ifstream input_labels("train-labels-idx1-ubyte", std::ios::binary);
	if (!input_labels.is_open()) {
		std::cout << "unable to open source file";
		return -1;
	}
	std::vector<unsigned char> target(std::istreambuf_iterator<char>(input_labels), {});
	input_labels.close();
	target.erase(target.begin(), std::next(target.begin(), size_target_header));

	int dim_data = train_data.size() / target.size();
	std::vector<std::vector<int>> masks_int(size_target, std::vector<int>(dim_data));
	std::vector<std::vector<float>> masks_float(size_target, std::vector<float>(dim_data));
	int masks_max[size_target] = { 0 };

	// mask generation

	int current_label = 0;
	int data_iterator = 0;
	for (int _data_point = 0; _data_point < target.size(); _data_point++) {
		current_label = target[_data_point];
		for (int _component = 0; _component < dim_data; _component++) {
			masks_int[current_label][_component] += train_data[data_iterator];
			if (masks_max[current_label] < masks_int[current_label][_component]) {
				masks_max[current_label] = masks_int[current_label][_component];
			}
			data_iterator++;
		}
	}

	for (int _data_point = 0; _data_point < size_target; _data_point++) {
		for (int _component = 0; _component < dim_data; _component++) {
			masks_float[_data_point][_component] =
				(float)masks_int[_data_point][_component] / (float)masks_max[_data_point];
		}
	}

	std::ofstream out_masks;
	out_masks.open("masks.txt");
	for (int _mask = 0; _mask < size_target; _mask++) {
		for (int _component = 0; _component < dim_data; _component++) {
			out_masks << masks_float[_mask][_component] << "; ";
		}
		out_masks << "\n";
	}
	out_masks.close();

	// data processing

	std::ofstream output_train_data("mean_digit_convolution_train_data.txt");

	data_iterator = 0;
	int current_id_sum_int = 0;
	float current_convolve = 0.0;
	float current_id_sum_float = 0.0;
	for (int _data_point = 0; _data_point < train_data.size() / dim_data; _data_point++) {

		current_id_sum_int = 0;
		for (int _component = 0; _component < dim_data; _component++) {
			current_id_sum_int += train_data[data_iterator];
			data_iterator++;
		}
		data_iterator -= dim_data;
		current_id_sum_float = (float)current_id_sum_int;

		for (int _mask_no = 0; _mask_no < masks_float.size(); _mask_no++) {
			current_convolve = 0.0;
			for (int _component = 0; _component < dim_data; _component++) {
				current_convolve += (float)train_data[data_iterator] * masks_float[_mask_no][_component];
				data_iterator++;
			}
			output_train_data << current_convolve / current_id_sum_float << ";";
			data_iterator -= dim_data;
		}
		output_train_data << "\n";
		data_iterator += dim_data;
	}

	output_train_data.close();

	std::ofstream output_test_data("mean_digit_convolution_test_data.txt");

	data_iterator = 0;
	current_id_sum_float = 0.0;
	for (int _data_point = 0; _data_point < test_data.size() / dim_data; _data_point++) {

		current_id_sum_int = 0;
		for (int _component = 0; _component < dim_data; _component++) {
			current_id_sum_int += test_data[data_iterator];
			data_iterator++;
		}
		data_iterator -= dim_data;
		current_id_sum_float = (float)current_id_sum_int;

		for (int _mask_no = 0; _mask_no < masks_float.size(); _mask_no++) {
			current_convolve = 0.0;
			for (int _component = 0; _component < dim_data; _component++) {
				current_convolve += (float)test_data[data_iterator] * masks_float[_mask_no][_component];
				data_iterator++;
			}
			output_test_data << current_convolve / current_id_sum_float << ";";
			data_iterator -= dim_data;
		}
		output_test_data << "\n";
		data_iterator += dim_data;
	}

	output_test_data.close();

	return 0;
}