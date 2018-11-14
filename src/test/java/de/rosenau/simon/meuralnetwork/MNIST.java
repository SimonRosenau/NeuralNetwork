package de.rosenau.simon.meuralnetwork;

import lombok.Getter;

import java.io.DataInputStream;
import java.io.FileInputStream;

/**
 * Project created by Simon Rosenau.
 */

@Getter
public class MNIST {

    private int numLabels;
    private int numImages;
    private int numRows;
    private int numCols;

    private byte[] labels;
    private double[][] data;

    public MNIST(String imageFile, String labelFile) throws Exception {
        DataInputStream labels = new DataInputStream(new FileInputStream(labelFile));
        DataInputStream images = new DataInputStream(new FileInputStream(imageFile));

        int magicNumber = labels.readInt();
        if (magicNumber != 2049) {
            throw new Exception("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
        }
        magicNumber = images.readInt();
        if (magicNumber != 2051) {
            throw new Exception("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
        }

        this.numLabels = labels.readInt();
        this.numImages = images.readInt();
        this.numRows = images.readInt();
        this.numCols = images.readInt();

        if (numLabels != numImages) {
            String str = "Image file and label file do not contain the same number of entries.\n" +
                    "  Label file contains: " + numLabels + "\n" +
                    "  Image file contains: " + numImages + "\n";
            throw new Exception(str);
        }

        this.labels = new byte[numLabels];
        labels.read(this.labels);

        int imageVectorSize = numCols * numRows;
        byte[] imagesData = new byte[numLabels * imageVectorSize];
        images.read(imagesData);

        this.data = new double[this.labels.length][imageVectorSize];

        for (int i = 0; i < this.labels.length; i++) {
            for (int o = 0; o < imageVectorSize; o++) {
                this.data[i][o] = (imagesData[i * imageVectorSize + o] & 0xff) / 255D;
            }
        }

        images.close();
        labels.close();
    }

}
