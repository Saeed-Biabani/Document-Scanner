from skimage import filters
import numpy as np
import cv2


def DrawBox(img, approx, to_rgb = False):
    if isinstance(approx, np.ndarray):
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.drawContours(img, [approx], -1, (0, 255, 255), 10)
    else:
        return img


def ResizeLike(mask, org):
    return cv2.resize(mask, org.shape[::-1], interpolation = cv2.INTER_LINEAR)


def ComputeRoiKeyPoints(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        
        peri = cv2.arcLength(cnts, True)
        approx = cv2.approxPolyDP(cnts, 0.02 * peri, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            
            rect = np.zeros((4, 2), dtype = "float32")
            
            s = pts.sum(axis = 1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            diff = np.diff(pts, axis = 1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            return rect, approx
    return None, None


def ComputeRoiShape(coords):
    (tl, tr, br, bl) = coords
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    return maxWidth, maxHeight


def ImagePerspective(img, shape, kpoints):
    maxWidth, maxHeight = shape

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(kpoints, dst)

    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def ExtractPaper(img, mask):
    kpoints, approx = ComputeRoiKeyPoints(mask)
    if isinstance(kpoints, np.ndarray):
        shape_ = ComputeRoiShape(kpoints)
        return ImagePerspective(img, shape_, kpoints), approx
    else:
        return None, None


def ScannSavedImage(fname, scanner, gray_paper = False):
    org = cv2.imread(fname)
    
    org_gray = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
    org_resize = cv2.resize(org_gray, (256, 256), interpolation = cv2.INTER_AREA)

    mask = scanner.ScanView(org_resize)
    
    mask = ResizeLike(mask, org_gray)
    
    if not gray_paper:
        org_gray = org
        
    paper, approx = ExtractPaper(org_gray, mask)

    org = DrawBox(org, approx)
    
    return paper, org


def EnhancePaper(img):
    if isinstance(img, np.ndarray):
        smooth = cv2.GaussianBlur(img, (95,95), 0)
        division = cv2.divide(img, smooth, scale=255)
        result = filters.unsharp_mask(division, radius=1.5, amount=1.5, preserve_range=False)
        return (255*result).clip(0,255).astype(np.uint8)
    return img


def SaveCompImage(fname, org, paper):
    size_ = org.shape
    base = np.zeros((size_[0], size_[1]*2, 3), dtype = "uint8") + 125
    base[0:size_[0], 0:size_[1]] = org

    offset = (np.array(size_[:2]) - np.array(paper.shape)) // 2
    base[offset[0]:offset[0]+paper.shape[0], size_[1]+offset[1]:size_[1]+offset[1]+paper.shape[1]] = cv2.cvtColor(paper, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(fname, base)