from matplotlib.colors import ListedColormap

tb_oleron_data =[
[0, 0, 2] ,
[0, 0, 2] ,
[0, 0, 2] ,
[0, 0, 2] ,
[0, 0, 2] ,
[0, 0, 2] ,
[0, 0, 3] ,
[0, 0, 3] ,
[0, 1, 3] ,
[0, 1, 3] ,
[0, 1, 4] ,
[0, 1, 4] ,
[0, 1, 4] ,
[0, 1, 4] ,
[1, 1, 5] ,
[1, 1, 5] ,
[1, 1, 5] ,
[1, 1, 5] ,
[1, 1, 6] ,
[1, 1, 6] ,
[1, 2, 6] ,
[1, 2, 7] ,
[1, 2, 7] ,
[1, 2, 7] ,
[1, 2, 8] ,
[1, 2, 8] ,
[1, 2, 9] ,
[1, 2, 9] ,
[1, 2, 9] ,
[2, 3, 10] ,
[2, 3, 10] ,
[2, 3, 11] ,
[2, 3, 11] ,
[2, 3, 12] ,
[2, 3, 12] ,
[2, 3, 12] ,
[2, 3, 13] ,
[2, 4, 13] ,
[2, 4, 14] ,
[2, 4, 14] ,
[2, 4, 15] ,
[2, 4, 15] ,
[3, 4, 15] ,
[3, 5, 16] ,
[3, 5, 16] ,
[3, 5, 17] ,
[3, 5, 17] ,
[3, 5, 18] ,
[3, 5, 18] ,
[3, 5, 18] ,
[3, 6, 19] ,
[3, 6, 19] ,
[4, 6, 20] ,
[4, 6, 20] ,
[4, 6, 21] ,
[4, 6, 21] ,
[4, 7, 21] ,
[4, 7, 22] ,
[4, 7, 22] ,
[4, 7, 23] ,
[4, 7, 23] ,
[4, 8, 24] ,
[5, 8, 24] ,
[5, 8, 25] ,
[5, 8, 25] ,
[5, 8, 25] ,
[5, 8, 26] ,
[5, 9, 26] ,
[5, 9, 27] ,
[5, 9, 27] ,
[5, 9, 28] ,
[5, 9, 28] ,
[6, 10, 28] ,
[6, 10, 29] ,
[6, 10, 29] ,
[6, 10, 30] ,
[6, 11, 30] ,
[6, 11, 31] ,
[6, 11, 31] ,
[6, 11, 32] ,
[6, 11, 32] ,
[7, 12, 33] ,
[7, 12, 33] ,
[7, 12, 33] ,
[7, 12, 34] ,
[7, 12, 34] ,
[7, 13, 35] ,
[7, 13, 35] ,
[7, 13, 36] ,
[8, 13, 36] ,
[8, 13, 37] ,
[8, 14, 37] ,
[8, 14, 38] ,
[8, 14, 38] ,
[8, 14, 39] ,
[8, 14, 39] ,
[8, 15, 39] ,
[9, 15, 40] ,
[9, 15, 40] ,
[9, 15, 41] ,
[9, 15, 41] ,
[9, 16, 42] ,
[9, 16, 42] ,
[9, 16, 43] ,
[9, 16, 43] ,
[10, 16, 44] ,
[10, 17, 44] ,
[10, 17, 45] ,
[10, 17, 45] ,
[10, 17, 46] ,
[10, 17, 46] ,
[10, 18, 47] ,
[11, 18, 47] ,
[11, 18, 48] ,
[11, 18, 48] ,
[11, 18, 49] ,
[11, 19, 49] ,
[11, 19, 50] ,
[11, 19, 50] ,
[12, 19, 51] ,
[12, 19, 51] ,
[12, 20, 52] ,
[12, 20, 52] ,
[12, 20, 53] ,
[12, 20, 53] ,
[13, 20, 54] ,
[13, 21, 54] ,
[13, 21, 55] ,
[13, 21, 55] ,
[13, 21, 56] ,
[13, 22, 56] ,
[13, 22, 57] ,
[14, 22, 57] ,
[14, 22, 58] ,
[14, 22, 58] ,
[14, 23, 59] ,
[14, 23, 59] ,
[14, 23, 60] ,
[15, 23, 60] ,
[15, 24, 61] ,
[15, 24, 61] ,
[15, 24, 62] ,
[15, 24, 62] ,
[15, 24, 63] ,
[16, 25, 63] ,
[16, 25, 64] ,
[16, 25, 64] ,
[16, 25, 65] ,
[16, 26, 65] ,
[17, 26, 66] ,
[17, 26, 66] ,
[17, 26, 67] ,
[17, 27, 67] ,
[17, 27, 68] ,
[17, 27, 68] ,
[18, 27, 68] ,
[18, 28, 69] ,
[18, 28, 69] ,
[18, 28, 70] ,
[18, 28, 70] ,
[19, 29, 71] ,
[19, 29, 71] ,
[19, 29, 72] ,
[19, 29, 72] ,
[19, 30, 73] ,
[20, 30, 73] ,
[20, 30, 74] ,
[20, 31, 74] ,
[20, 31, 75] ,
[21, 31, 75] ,
[21, 31, 75] ,
[21, 32, 76] ,
[21, 32, 76] ,
[21, 32, 77] ,
[22, 32, 77] ,
[22, 33, 78] ,
[22, 33, 78] ,
[22, 33, 78] ,
[23, 34, 79] ,
[23, 34, 79] ,
[23, 34, 80] ,
[23, 35, 80] ,
[23, 35, 80] ,
[24, 35, 81] ,
[24, 36, 81] ,
[24, 36, 81] ,
[24, 36, 82] ,
[25, 37, 82] ,
[25, 37, 82] ,
[25, 37, 82] ,
[25, 38, 83] ,
[26, 38, 83] ,
[26, 38, 83] ,
[26, 39, 83] ,
[26, 39, 83] ,
[26, 39, 83] ,
[27, 39, 84] ,
[27, 40, 84] ,
[27, 40, 84] ,
[27, 40, 84] ,
[27, 41, 84] ,
[27, 41, 84] ,
[28, 41, 84] ,
[28, 42, 84] ,
[28, 42, 84] ,
[28, 42, 84] ,
[28, 42, 83] ,
[28, 43, 83] ,
[29, 43, 83] ,
[29, 43, 83] ,
[29, 43, 83] ,
[29, 44, 83] ,
[29, 44, 83] ,
[29, 44, 83] ,
[29, 44, 82] ,
[29, 45, 82] ,
[30, 45, 82] ,
[30, 45, 82] ,
[30, 45, 82] ,
[30, 45, 82] ,
[30, 46, 81] ,
[30, 46, 81] ,
[30, 46, 81] ,
[30, 46, 81] ,
[30, 46, 81] ,
[31, 47, 80] ,
[31, 47, 80] ,
[31, 47, 80] ,
[31, 47, 80] ,
[31, 47, 80] ,
[31, 47, 79] ,
[31, 48, 79] ,
[31, 48, 79] ,
[31, 48, 79] ,
[31, 48, 79] ,
[31, 48, 78] ,
[32, 48, 78] ,
[32, 48, 78] ,
[32, 49, 78] ,
[32, 49, 78] ,
[32, 49, 77] ,
[32, 49, 77] ,
[32, 49, 77] ,
[32, 49, 77] ,
[32, 50, 77] ,
[32, 50, 76] ,
[32, 50, 76] ,
[33, 50, 76] ,
[33, 50, 76] ,
[33, 50, 76] ,
[33, 50, 75] ,
[33, 50, 75] ,
[33, 51, 75] ,
[33, 51, 75] ,
[33, 51, 75] ,
[33, 51, 74] ,
[33, 51, 74] ,
[33, 51, 74] ,
[34, 51, 74] ,
[34, 51, 74] ,
[34, 52, 73] ,
[34, 52, 73] ,
[34, 52, 73] ,
[34, 52, 73] ,
[34, 52, 73] ,
[34, 52, 73] ,
[34, 52, 72] ,
[34, 52, 72] ,
[34, 52, 72] ,
[34, 53, 72] ,
[35, 53, 72] ,
[35, 53, 71] ,
[35, 53, 71] ,
[35, 53, 71] ,
[35, 53, 71] ,
[35, 53, 71] ,
[35, 53, 71] ,
[35, 53, 70] ,
[35, 54, 70] ,
[35, 54, 70] ,
[35, 54, 70] ,
[35, 54, 70] ,
[35, 54, 70] ,
[36, 54, 69] ,
[36, 54, 69] ,
[36, 54, 69] ,
[36, 54, 69] ,
[36, 54, 69] ,
[36, 55, 69] ,
[36, 55, 68] ,
[36, 55, 68] ,
[36, 55, 68] ,
[36, 55, 68] ,
[36, 55, 68] ,
[36, 55, 68] ,
[36, 55, 68] ,
[36, 55, 67] ,
[37, 55, 67] ,
[37, 56, 67] ,
[37, 56, 67] ,
[37, 56, 67] ,
[37, 56, 67] ,
[37, 56, 67] ,
[37, 56, 66] ,
[37, 56, 66] ,
[37, 56, 66] ,
[37, 56, 66] ,
[37, 56, 66] ,
[37, 56, 66] ,
[37, 57, 66] ,
[37, 57, 65] ,
[37, 57, 65] ,
[37, 57, 65] ,
[38, 57, 65] ,
[38, 57, 65] ,
[38, 57, 65] ,
[38, 57, 65] ,
[38, 57, 64] ,
[38, 57, 64] ,
[38, 58, 64] ,
[38, 58, 64] ,
[38, 58, 64] ,
[38, 58, 64] ,
[38, 58, 64] ,
[38, 58, 64] ,
[38, 58, 63] ,
[38, 58, 63] ,
[38, 58, 63] ,
[38, 58, 63] ,
[38, 58, 63] ,
[38, 59, 63] ,
[38, 59, 63] ,
[38, 59, 63] ,
[38, 59, 62] ,
[38, 59, 62] ,
[38, 59, 62] ,
[38, 59, 62] ,
[38, 59, 62] ,
[38, 59, 62] ,
[39, 59, 62] ,
[39, 59, 62] ,
[39, 60, 62] ,
[39, 60, 61] ,
[39, 60, 61] ,
[39, 60, 61] ,
[39, 60, 61] ,
[39, 60, 61] ,
[39, 60, 61] ,
[39, 60, 61] ,
[39, 60, 61] ,
[39, 60, 60] ,
[39, 61, 60] ,
[39, 61, 60] ,
[39, 61, 60] ,
[39, 61, 60] ,
[39, 61, 60] ,
[39, 61, 60] ,
[39, 61, 60] ,
[39, 61, 59] ,
[39, 61, 59] ,
[39, 62, 59] ,
[39, 62, 59] ,
[39, 62, 59] ,
[38, 62, 59] ,
[38, 62, 59] ,
[38, 62, 59] ,
[38, 62, 58] ,
[38, 62, 58] ,
[38, 62, 58] ,
[38, 62, 58] ,
[38, 63, 58] ,
[38, 63, 58] ,
[38, 63, 58] ,
[38, 63, 58] ,
[38, 63, 57] ,
[38, 63, 57] ,
[38, 63, 57] ,
[38, 63, 57] ,
[38, 63, 57] ,
[38, 64, 57] ,
[38, 64, 57] ,
[38, 64, 56] ,
[38, 64, 56] ,
[38, 64, 56] ,
[38, 64, 56] ,
[38, 64, 56] ,
[38, 64, 56] ,
[38, 64, 55] ,
[38, 65, 55] ,
[38, 65, 55] ,
[38, 65, 55] ,
[37, 65, 55] ,
[37, 65, 55] ,
[37, 65, 54] ,
[37, 65, 54] ,
[37, 65, 54] ,
[37, 65, 54] ,
[37, 66, 54] ,
[37, 66, 54] ,
[37, 66, 53] ,
[37, 66, 53] ,
[37, 66, 53] ,
[37, 66, 53] ,
[37, 66, 53] ,
[37, 66, 53] ,
[37, 66, 52] ,
[37, 67, 52] ,
[37, 67, 52] ,
[37, 67, 52] ,
[37, 67, 52] ,
[37, 67, 51] ,
[36, 67, 51] ,
[36, 67, 51] ,
[36, 67, 51] ,
[36, 68, 51] ,
[36, 68, 50] ,
[36, 68, 50] ,
[36, 68, 50] ,
[36, 68, 50] ,
[36, 68, 50] ,
[36, 68, 49] ,
[36, 68, 49] ,
[36, 68, 49] ,
[36, 69, 49] ,
[36, 69, 48] ,
[36, 69, 48] ,
[36, 69, 48] ,
[36, 69, 48] ,
[36, 69, 48] ,
[36, 69, 47] ,
[35, 69, 47] ,
[35, 69, 47] ,
[35, 70, 47] ,
[35, 70, 46] ,
[35, 70, 46] ,
[35, 70, 46] ,
[35, 70, 46] ,
[35, 70, 46] ,
[35, 70, 45] ,
[35, 70, 45] ,
[35, 71, 45] ,
[35, 71, 45] ,
[35, 71, 44] ,
[35, 71, 44] ,
[35, 71, 44] ,
[35, 71, 44] ,
[35, 71, 43] ,
[35, 71, 43] ,
[35, 72, 43] ,
[35, 72, 43] ,
[35, 72, 42] ,
[35, 72, 42] ,
[34, 72, 42] ,
[34, 72, 42] ,
[34, 72, 41] ,
[34, 72, 41] ,
[34, 72, 41] ,
[34, 73, 40] ,
[34, 73, 40] ,
[34, 73, 40] ,
[34, 73, 40] ,
[34, 73, 39] ,
[34, 73, 39] ,
[34, 73, 39] ,
[34, 73, 39] ,
[34, 74, 38] ,
[34, 74, 38] ,
[34, 74, 38] ,
[34, 74, 37] ,
[34, 74, 37] ,
[34, 74, 37] ,
[34, 74, 37] ,
[34, 74, 36] ,
[34, 75, 36] ,
[34, 75, 36] ,
[34, 75, 35] ,
[34, 75, 35] ,
[34, 75, 35] ,
[34, 75, 34] ,
[34, 75, 34] ,
[34, 75, 34] ,
[34, 76, 33] ,
[34, 76, 33] ,
[34, 76, 33] ,
[34, 76, 33] ,
[34, 76, 32] ,
[34, 76, 32] ,
[34, 76, 32] ,
[34, 77, 31] ,
[34, 77, 31] ,
[34, 77, 31] ,
[34, 77, 30] ,
[34, 77, 30] ,
[34, 77, 30] ,
[34, 77, 29] ,
[34, 77, 29] ,
[34, 78, 29] ,
[34, 78, 28] ,
[34, 78, 28] ,
[34, 78, 28] ,
[34, 78, 27] ,
[34, 78, 27] ,
[34, 78, 27] ,
[34, 79, 26] ,
[34, 79, 26] ,
[34, 79, 26] ,
[34, 79, 25] ,
[34, 79, 25] ,
[34, 79, 24] ,
[34, 80, 24] ,
[34, 80, 24] ,
[34, 80, 23] ,
[34, 80, 23] ,
[35, 80, 23] ,
[35, 80, 22] ,
[35, 81, 22] ,
[35, 81, 22] ,
[35, 81, 21] ,
[35, 81, 21] ,
[35, 81, 21] ,
[35, 82, 21] ,
[36, 82, 20] ,
[36, 82, 20] ,
[36, 82, 20] ,
[36, 83, 20] ,
[36, 83, 20] ,
[37, 83, 20] ,
[37, 83, 19] ,
[37, 84, 19] ,
[38, 84, 19] ,
[38, 84, 20] ,
[38, 85, 20] ,
[39, 85, 20] ,
[39, 85, 20] ,
[39, 85, 20] ,
[40, 86, 20] ,
[40, 86, 21] ,
[40, 86, 21] ,
[41, 86, 21] ,
[41, 87, 21] ,
[42, 87, 22] ,
[42, 87, 22] ,
[42, 88, 22] ,
[43, 88, 22] ,
[43, 88, 23] ,
[43, 88, 23] ,
[44, 89, 23] ,
[44, 89, 24] ,
[44, 89, 24] ,
[45, 89, 24] ,
[45, 90, 25] ,
[46, 90, 25] ,
[46, 90, 25] ,
[46, 90, 26] ,
[47, 91, 26] ,
[47, 91, 26] ,
[47, 91, 27] ,
[48, 92, 27] ,
[48, 92, 27] ,
[48, 92, 28] ,
[49, 92, 28] ,
[49, 93, 28] ,
[49, 93, 29] ,
[50, 93, 29] ,
[50, 93, 29] ,
[50, 94, 30] ,
[51, 94, 30] ,
[51, 94, 30] ,
[52, 94, 31] ,
[52, 95, 31] ,
[52, 95, 31] ,
[53, 95, 32] ,
[53, 95, 32] ,
[53, 96, 32] ,
[54, 96, 33] ,
[54, 96, 33] ,
[54, 96, 33] ,
[55, 97, 34] ,
[55, 97, 34] ,
[55, 97, 34] ,
[56, 97, 35] ,
[56, 98, 35] ,
[56, 98, 35] ,
[57, 98, 36] ,
[57, 99, 36] ,
[57, 99, 36] ,
[58, 99, 37] ,
[58, 99, 37] ,
[59, 100, 37] ,
[59, 100, 38] ,
[59, 100, 38] ,
[60, 100, 38] ,
[60, 101, 39] ,
[60, 101, 39] ,
[61, 101, 39] ,
[61, 101, 40] ,
[61, 102, 40] ,
[62, 102, 40] ,
[62, 102, 41] ,
[62, 102, 41] ,
[63, 103, 41] ,
[63, 103, 42] ,
[63, 103, 42] ,
[64, 103, 42] ,
[64, 104, 43] ,
[64, 104, 43] ,
[65, 104, 43] ,
[65, 104, 44] ,
[65, 105, 44] ,
[66, 105, 44] ,
[66, 105, 45] ,
[66, 106, 45] ,
[67, 106, 45] ,
[67, 106, 46] ,
[68, 106, 46] ,
[68, 107, 46] ,
[68, 107, 47] ,
[69, 107, 47] ,
[69, 107, 47] ,
[69, 108, 48] ,
[70, 108, 48] ,
[70, 108, 48] ,
[70, 108, 49] ,
[71, 109, 49] ,
[71, 109, 49] ,
[71, 109, 50] ,
[72, 109, 50] ,
[72, 110, 50] ,
[72, 110, 51] ,
[73, 110, 51] ,
[73, 111, 51] ,
[73, 111, 52] ,
[74, 111, 52] ,
[74, 111, 52] ,
[74, 112, 53] ,
[75, 112, 53] ,
[75, 112, 53] ,
[75, 112, 54] ,
[76, 113, 54] ,
[76, 113, 54] ,
[76, 113, 55] ,
[77, 113, 55] ,
[77, 114, 55] ,
[78, 114, 56] ,
[78, 114, 56] ,
[78, 114, 56] ,
[79, 115, 57] ,
[79, 115, 57] ,
[79, 115, 57] ,
[80, 115, 58] ,
[80, 116, 58] ,
[80, 116, 58] ,
[81, 116, 59] ,
[81, 117, 59] ,
[81, 117, 59] ,
[82, 117, 60] ,
[82, 117, 60] ,
[82, 118, 60] ,
[83, 118, 61] ,
[83, 118, 61] ,
[83, 118, 61] ,
[84, 119, 62] ,
[84, 119, 62] ,
[84, 119, 62] ,
[85, 119, 63] ,
[85, 120, 63] ,
[85, 120, 63] ,
[86, 120, 64] ,
[86, 121, 64] ,
[87, 121, 64] ,
[87, 121, 65] ,
[87, 121, 65] ,
[88, 122, 65] ,
[88, 122, 66] ,
[88, 122, 66] ,
[89, 122, 66] ,
[89, 123, 67] ,
[89, 123, 67] ,
[90, 123, 68] ,
[90, 123, 68] ,
[90, 124, 68] ,
[91, 124, 69] ,
[91, 124, 69] ,
[91, 125, 69] ,
[92, 125, 70] ,
[92, 125, 70] ,
[92, 125, 70] ,
[93, 126, 71] ,
[93, 126, 71] ,
[93, 126, 71] ,
[94, 126, 72] ,
[94, 127, 72] ,
[95, 127, 72] ,
[95, 127, 73] ,
[95, 127, 73] ,
[96, 128, 73] ,
[96, 128, 74] ,
[96, 128, 74] ,
[97, 129, 74] ,
[97, 129, 75] ,
[97, 129, 75] ,
[98, 129, 75] ,
[98, 130, 76] ,
[98, 130, 76] ,
[99, 130, 76] ,
[99, 130, 77] ,
[99, 131, 77] ,
[100, 131, 78] ,
[100, 131, 78] ,
[101, 132, 78] ,
[101, 132, 79] ,
[101, 132, 79] ,
[102, 132, 79] ,
[102, 133, 80] ,
[102, 133, 80] ,
[103, 133, 80] ,
[103, 133, 81] ,
[103, 134, 81] ,
[104, 134, 81] ,
[104, 134, 82] ,
[104, 134, 82] ,
[105, 135, 82] ,
[105, 135, 83] ,
[105, 135, 83] ,
[106, 136, 83] ,
[106, 136, 84] ,
[107, 136, 84] ,
[107, 136, 85] ,
[107, 137, 85] ,
[108, 137, 85] ,
[108, 137, 86] ,
[108, 138, 86] ,
[109, 138, 86] ,
[109, 138, 87] ,
[109, 138, 87] ,
[110, 139, 87] ,
[110, 139, 88] ,
[110, 139, 88] ,
[111, 139, 88] ,
[111, 140, 89] ,
[112, 140, 89] ,
[112, 140, 89] ,
[112, 141, 90] ,
[113, 141, 90] ,
[113, 141, 91] ,
[113, 141, 91] ,
[114, 142, 91] ,
[114, 142, 92] ,
[114, 142, 92] ,
[115, 142, 92] ,
[115, 143, 93] ,
[115, 143, 93] ,
[116, 143, 93] ,
[116, 144, 94] ,
[117, 144, 94] ,
[117, 144, 94] ,
[117, 144, 95] ,
[118, 145, 95] ,
[118, 145, 96] ,
[118, 145, 96] ,
[119, 146, 96] ,
[119, 146, 97] ,
[119, 146, 97] ,
[120, 146, 97] ,
[120, 147, 98] ,
[121, 147, 98] ,
[121, 147, 98] ,
[121, 147, 99] ,
[122, 148, 99] ,
[122, 148, 99] ,
[122, 148, 100] ,
[123, 149, 100] ,
[123, 149, 101] ,
[123, 149, 101] ,
[124, 149, 101] ,
[124, 150, 102] ,
[125, 150, 102] ,
[125, 150, 102] ,
[125, 151, 103] ,
[126, 151, 103] ,
[126, 151, 103] ,
[126, 151, 104] ,
[127, 152, 104] ,
[127, 152, 105] ,
[127, 152, 105] ,
[128, 153, 105] ,
[128, 153, 106] ,
[129, 153, 106] ,
[129, 153, 106] ,
[129, 154, 107] ,
[130, 154, 107] ,
[130, 154, 108] ,
[130, 155, 108] ,
[131, 155, 108] ,
[131, 155, 109] ,
[132, 155, 109] ,
[132, 156, 109] ,
[132, 156, 110] ,
[133, 156, 110] ,
[133, 157, 110] ,
[133, 157, 111] ,
[134, 157, 111] ,
[134, 157, 112] ,
[134, 158, 112] ,
[135, 158, 112] ,
[135, 158, 113] ,
[136, 159, 113] ,
[136, 159, 113] ,
[136, 159, 114] ,
[137, 159, 114] ,
[137, 160, 115] ,
[137, 160, 115] ,
[138, 160, 115] ,
[138, 161, 116] ,
[139, 161, 116] ,
[139, 161, 116] ,
[139, 162, 117] ,
[140, 162, 117] ,
[140, 162, 118] ,
[140, 162, 118] ,
[141, 163, 118] ,
[141, 163, 119] ,
[142, 163, 119] ,
[142, 164, 119] ,
[142, 164, 120] ,
[143, 164, 120] ,
[143, 164, 121] ,
[143, 165, 121] ,
[144, 165, 121] ,
[144, 165, 122] ,
[145, 166, 122] ,
[145, 166, 122] ,
[145, 166, 123] ,
[146, 167, 123] ,
[146, 167, 124] ,
[147, 167, 124] ,
[147, 167, 124] ,
[147, 168, 125] ,
[148, 168, 125] ,
[148, 168, 126] ,
[148, 169, 126] ,
[149, 169, 126] ,
[149, 169, 127] ,
[150, 170, 127] ,
[150, 170, 127] ,
[150, 170, 128] ,
[151, 170, 128] ,
[151, 171, 129] ,
[151, 171, 129] ,
[152, 171, 129] ,
[152, 172, 130] ,
[153, 172, 130] ,
[153, 172, 130] ,
[153, 173, 131] ,
[154, 173, 131] ,
[154, 173, 132] ,
[155, 173, 132] ,
[155, 174, 132] ,
[155, 174, 133] ,
[156, 174, 133] ,
[156, 175, 134] ,
[157, 175, 134] ,
[157, 175, 134] ,
[157, 176, 135] ,
[158, 176, 135] ,
[158, 176, 136] ,
[158, 176, 136] ,
[159, 177, 136] ,
[159, 177, 137] ,
[160, 177, 137] ,
[160, 178, 137] ,
[160, 178, 138] ,
[161, 178, 138] ,
[161, 179, 139] ,
[162, 179, 139] ,
[162, 179, 139] ,
[162, 180, 140] ,
[163, 180, 140] ,
[163, 180, 141] ,
[164, 180, 141] ,
[164, 181, 141] ,
[164, 181, 142] ,
[165, 181, 142] ,
[165, 182, 143] ,
[166, 182, 143] ,
[166, 182, 143] ,
[166, 183, 144] ,
[167, 183, 144] ,
[167, 183, 145] ,
[167, 184, 145] ,
[168, 184, 145] ,
[168, 184, 146] ,
[169, 185, 146] ,
[169, 185, 147] ,
[169, 185, 147] ,
[170, 185, 147] ,
[170, 186, 148] ,
[171, 186, 148] ,
[171, 186, 149] ,
[171, 187, 149] ,
[172, 187, 149] ,
[172, 187, 150] ,
[173, 188, 150] ,
[173, 188, 151] ,
[173, 188, 151] ,
[174, 189, 151] ,
[174, 189, 152] ,
[175, 189, 152] ,
[175, 190, 153] ,
[175, 190, 153] ,
[176, 190, 153] ,
[176, 191, 154] ,
[177, 191, 154] ,
[177, 191, 155] ,
[178, 191, 155] ,
[178, 192, 155] ,
[178, 192, 156] ,
[179, 192, 156] ,
[179, 193, 157] ,
[180, 193, 157] ,
[180, 193, 158] ,
[180, 194, 158] ,
[181, 194, 158] ,
[181, 194, 159] ,
[182, 195, 159] ,
[182, 195, 160] ,
[182, 195, 160] ,
[183, 196, 160] ,
[183, 196, 161] ,
[184, 196, 161] ,
[184, 197, 162] ,
[184, 197, 162] ,
[185, 197, 162] ,
[185, 198, 163] ,
[186, 198, 163] ,
[186, 198, 164] ,
[187, 199, 164] ,
[187, 199, 165] ,
[187, 199, 165] ,
[188, 200, 165] ,
[188, 200, 166] ,
[189, 200, 166] ,
[189, 201, 167] ,
[189, 201, 167] ,
[190, 201, 167] ,
[190, 202, 168] ,
[191, 202, 168] ,
[191, 202, 169] ,
[192, 203, 169] ,
[192, 203, 170] ,
[192, 203, 170] ,
[193, 204, 170] ,
[193, 204, 171] ,
[194, 204, 171] ,
[194, 205, 172] ,
[194, 205, 172] ,
[195, 205, 172] ,
[195, 206, 173] ,
[196, 206, 173] ,
[196, 206, 174] ,
[197, 207, 174] ,
[197, 207, 175] ,
[197, 207, 175] ,
[198, 208, 175] ,
[198, 208, 176] ,
[199, 208, 176] ,
[199, 209, 177] ,
[200, 209, 177] ,
[200, 209, 178] ,
[200, 210, 178] ,
[201, 210, 178] ,
[201, 210, 179] ,
[202, 211, 179] ,
[202, 211, 180] ,
[203, 211, 180] ,
[203, 212, 181] ,
[203, 212, 181] ,
[204, 213, 181] ,
[204, 213, 182] ,
[205, 213, 182] ,
[205, 214, 183] ,
[206, 214, 183] ,
[206, 214, 184] ,
[206, 215, 184] ,
[207, 215, 185] ,
[207, 215, 185] ,
[208, 216, 185] ,
[208, 216, 186] ,
[209, 216, 186] ,
[209, 217, 187] ,
[209, 217, 187] ,
[210, 217, 188] ,
[210, 218, 188] ,
[211, 218, 188] ,
[211, 218, 189] ,
[212, 219, 189] ,
[212, 219, 190] ,
[213, 220, 190] ,
[213, 220, 191] ,
[213, 220, 191] ,
[214, 221, 192] ,
[214, 221, 192] ,
[215, 221, 192] ,
[215, 222, 193] ,
[216, 222, 193] ,
[216, 222, 194] ,
[217, 223, 194] ,
[217, 223, 195] ,
[217, 223, 195] ,
[218, 224, 196] ,
[218, 224, 196] ,
[219, 225, 196] ,
[219, 225, 197] ,
[220, 225, 197] ,
[220, 226, 198] ,
[221, 226, 198] ,
[221, 226, 199] ,
[221, 227, 199] ,
[222, 227, 200] ,
[222, 227, 200] ,
[223, 228, 201] ,
[223, 228, 201] ,
[224, 229, 201] ,
[224, 229, 202] ]

tb_oleron = ListedColormap([[c[0]/255, c[1]/255, c[2]/255] for c in tb_oleron_data], name='tb_oleron')

