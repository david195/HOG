/*
 * David Ruiz García.
 *
*/

/*Step 1
   Sea aplica una mascara, usando la derivada de sobel.

  Step 2
   Se calcula el histograma de gradientes.
   Devuelve una matriz unidimencional.
   histogram_gradient(src,cell_TAM,bin_TAM);

 Step 4 se genera el vector caracteristico.
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "iostream"

using namespace std;
using namespace cv;

Mat getvec(Mat src,int nbin){
  /*Step 1*/
  Mat grad;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_32F;
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  /// Gradient X
  //Scharr( src, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  /// Gradient Y
  //Scharr( src, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  /// Total Gradient (approximate)

  /*Step 2*/
  Mat angle,magnitude;
  cartToPolar(grad_x, grad_y, magnitude,angle,1);
  /*Transform to unsigned gradient*/
  for(int i=0; i<angle.size().height;i++)
    for(int j=0; j<angle.size().width;j++){
      float intensity = angle.at<float>(i, j);
      if(intensity>180)
        angle.at<float>(Point(j,i)) = intensity-180;
    }
  int bin_tam = 180/nbin;
  Mat bin = Mat::zeros( nbin,1, CV_32FC1 );
  for(int x=0; x<magnitude.size().width; x++){
    for(int y=0; y<magnitude.size().height; y++){
      float intensity = angle.at<float>(y,x);
      int i=0;
      int val=0;
      do{
        if(intensity>=val && intensity < (val+bin_tam)){
          float mag = magnitude.at<float>(y,x)/255;
          if(intensity>=val && intensity<=val+90/bin_tam){
            bin.at<float>(i) += mag;
          }
          else{
            bin.at<float>(i+1) += mag;
          }
        }
        val+=bin_tam;
        i++;
      }while(i < nbin-1);
      float mag = magnitude.at<float>(y,x)/255;
      if(intensity>=val && intensity<=val+180/bin_tam){
        if(intensity>=val && intensity <=val+90/bin_tam)
          bin.at<float>(i) += mag;
        else
        bin.at<float>(0) += mag;
      }
    }
  }
  Mat bin_normalize;
  normalize(bin,bin_normalize,1,0, NORM_MINMAX);
  return bin_normalize;
}

Mat hog(Mat src, int cell_tam, int nbin){
  src.convertTo(src, CV_32F, 1/255.0);
  int nc = int (src.size().height/cell_tam) * int (src.size().width/cell_tam) * nbin;
  Mat vec = Mat::zeros( nc,1, CV_32FC1 );
  int x=0, y=0, nfeature=0;
  while(y<=(src.size().height-cell_tam)){
    x=0;
    while(x<=(src.size().width-cell_tam)){
      Mat roi(src, Rect(x,y,cell_tam,cell_tam));
      Mat cell = getvec(roi,nbin);
      for(int i=0;i<nbin;i++){
        float val = cell.at<float>(i);
        vec.at<float>(nfeature) = val;
        nfeature++;
      }
      x+=cell_tam;
    }
    y+=cell_tam;
  }
  return vec;
}

void view(Mat src, Mat vec, int cell_tam, int bin_tam){
  int nx = src.size().width/cell_tam;
  int ny = src.size().height/cell_tam;
  for(int i=0; i<nx; i++){
    for(int j=0; j<ny; j++){
      int x= i*cell_tam+cell_tam/4;
      int y= j*cell_tam+cell_tam/4;
      int m = (i*ny + j)*bin_tam;
      for(int aux=0;aux<bin_tam;aux++){
        float ang = CV_PI*(bin_tam * aux);
        float mag = cell_tam/4*vec.at<float>(m+aux);
        int sx = x+(mag)*cos(ang/180);
        int sy = y+(mag)*sin(ang/180);
        line(src, Point(x,y), Point(sx,sy), 255);
      }
      /*float max=0;
      int ind=0;
      for(int aux=0;aux<bin_tam;aux++){
        if(max < vec.at<float>(m+aux)){
          max = vec.at<float>(m+aux);
          ind = aux;
        }
      }
      float ang = 180/bin_tam * ind;
      float mag = max * cell_tam/2;
      int sx = x+(mag)*cos(ang);
      int sy = y+(mag)*sin(ang);
      line(src, Point(x,y), Point(sx,sy), 255);*/
    }
  }
  imshow("HOG",src);
}

/*void writeCSV(Mat vec, string dir){
  ostringstream csv;
  for(int i=0; i<vec.size().height; i++){
    csv << vec.at<float>(i) << ",";
  }
  csv << endl;
  ofstream file(dir);
  file << csv.str() << endl;
  file.close();
}*/

int main( int argc, char** argv ){
 Mat src;
 src = imread( argv[1] ,CV_LOAD_IMAGE_GRAYSCALE);
 if(argc < 4){ cout << "./main img.jpg cell_tam bin_tam" <<endl; return -1;}
 if( !src.data ){ cout << "Image not found" <<endl;return -1; }

 int ncell=stoi (argv[2],nullptr,10), nbin=stoi (argv[3],nullptr,10);
 Mat vec = hog(src,ncell,nbin);

 view(src,vec,ncell,nbin);
 cout << vec << endl << endl;
 cout << "S: " <<  vec.size().height << endl;
 waitKey(0);
 return 0;
 }
