import cv2
import numpy as np

# 读取mask图像
mask_image = cv2.imread(r'C:\Users\27726\Desktop\pic\img_out\1_out.jpg', 0)  # 以灰度图像方式读取

# 二值化mask图像
_, binary_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

# 查找不连通区域
#stata面积
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

# 创建一个彩色图像用于绘制轮廓
#contour_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
contour_image = cv2.imread(r'C:\Users\27726\Desktop\pic\xin\1_out.jpg')

# 遍历每个不连通区域
for label in range(1, num_labels):
    # 获取当前区域的面积
    area = stats[label, cv2.CC_STAT_AREA]

    # 获取当前区域的轮廓
    contour, _ = cv2.findContours((labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制当前区域的轮廓
    cv2.drawContours(contour_image, contour, -1, (0, 0, 255), 2)

    # # 在图像上显示当前区域的面积
    # cv2.putText(contour_image, f"Area: {area}", (int(centroids[label, 0]), int(centroids[label, 1])),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

# 显示结果图像
cv2.imwrite(r"C:\Users\27726\Desktop\pic\xin\1_out_s.jpg", contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()