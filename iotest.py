from pydicom import dcmread
def extractInfo(ds):
    info = {}
    info["PatientID"] = ds.PatientID
    info["PatientName"] = ds.PatientName
    info["PatientAge"] = ds.PatientAge
    info["PatientSex"] = ds.PatientSex
    info["StudyID"] = ds.StudyID
    info["StudyDate"] = ds.StudyDate
    info["StudyTime"] = ds.StudyTime
    return info
def main():
    path = "C:/Users/57035/Desktop/ELEC/3rd year/RSNA Chest Xray/images in initial annotation/1.2.276.0.7230010.3.1.2.8323329.10002.1517874346.165591/1.2.276.0.7230010.3.1.3.8323329.10002.1517874346.165590/1.2.276.0.7230010.3.1.4.8323329.10002.1517874346.165592.dcm"
    ds = dcmread(path)
    '''
    print(ds)
    print(type(ds.PixelData))
    print(len(ds.PixelData))
    print(ds.PixelData[:2])
    '''
    arr = ds.pixel_array
    print("the array is in the shape of", arr.shape)
    max = arr.max()
    min = arr.min()
    print("The maximum pixel is {0} and the minium is {1}".format(max,min))

if __name__ == "__main__":
    main()
