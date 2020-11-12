package com.example.tensor

import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

class Tensor {

    private lateinit var interpreter: Interpreter

    fun setInterpreter(mappedByteBuffer: MappedByteBuffer) {
        interpreter =
            Interpreter(
                mappedByteBuffer,
                Interpreter.Options().apply {
                    setNumThreads(4)
                    addDelegate(GpuDelegate())
                    setAllowBufferHandleOutput(false)
                }
            )
    }

    fun segmentImage(bitmap: Bitmap): ByteBuffer {

        val resizeBitmap = Bitmap.createScaledBitmap(
            bitmap,
            IMAGE_SIZE,
            IMAGE_SIZE, true
        )

        val segmentationMasks =
            ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * NUM_CLASSES * TO_FLOAT)

        segmentationMasks.order(ByteOrder.nativeOrder())

        interpreter.run(
            bitmapToByteBuffer(
                resizeBitmap,
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            segmentationMasks
        )
        return segmentationMasks
    }

    private fun bitmapToByteBuffer(
        bitmapIn: Bitmap,
        width: Int,
        height: Int,
        mean: Float = 0.0f,
        std: Float = 255.0f
    ): ByteBuffer {
        val inputImage = ByteBuffer.allocateDirect(1 * width * height * 3 * 4)
        inputImage.order(ByteOrder.nativeOrder())
        inputImage.rewind()

        val intValues = IntArray(width * height)
        bitmapIn.getPixels(intValues, 0, width, 0, 0, width, height)
        var pixel = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val value = intValues[pixel++]
                // Normalize channel values to [-1.0, 1.0]. This requirement varies by
                // model. For example, some models might require values to be normalized
                // to the range [0.0, 1.0] instead.
                inputImage.putFloat(((value shr 16 and 0xFF) - mean) / std)
                inputImage.putFloat(((value shr 8 and 0xFF) - mean) / std)
                inputImage.putFloat(((value and 0xFF) - mean) / std)
            }
        }
        inputImage.rewind()
        return inputImage
    }

    companion object {

        const val NUM_CLASSES = 21
        const val IMAGE_SIZE = 257
        const val TO_FLOAT = 4
    }

}