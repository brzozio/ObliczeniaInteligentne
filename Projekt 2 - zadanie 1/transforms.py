import numpy as np


def precompute_masks():
    mask = np.zeros((15, 28 * 28), dtype=np.float64)
    center_offset = 0.5 - 28.0 / 2.0
    for y in range(28):
        ty = y + center_offset
        ty_abs = np.abs(ty)
        ty_sgn = np.sign(ty)
        for x in range(28):
            tx = x + center_offset
            tx_abs = np.abs(tx)
            tx_sgn = np.sign(tx)
            position = 28 * y + x
            # positive definite radial and signed vertical, horizontal metrics
            # 2 metric
            mask[0, position] = np.sqrt(ty * ty + tx * tx)
            mask[1, position] = ty_sgn * ty * ty
            mask[2, position] = tx_sgn * tx * tx
            # 1 metric
            mask[3, position] = ty_abs + tx_abs
            mask[4, position] = ty
            mask[5, position] = tx
            # 0 metric
            mask[6, position] = ty_sgn * tx_sgn
            mask[7, position] = ty_sgn
            mask[8, position] = tx_sgn
            # -1 metric
            mask[9, position] = 1.0 / (1.0 / ty_abs + 1.0 / tx_abs)
            mask[10, position] = 1.0 / ty
            mask[11, position] = 1.0 / tx
            # inverse square root
            mask[12, position] = 1.0 / mask[0, position]
            # normalized units
            mask[13, position] = ty * mask[10, position]
            mask[14, position] = tx * mask[10, position]

    np.savetxt("masks_snake.csv", mask, delimiter=";")


def extract_labels(data_point: np.array, masks: np.array) -> np.array:
    output: np.array = np.zeros(15,dtype=np.float64)
    for mask_no in range(15):
        for pixel in range(28*28):
            output[mask_no] += data_point[pixel]*masks[mask_no, pixel]
    return output
