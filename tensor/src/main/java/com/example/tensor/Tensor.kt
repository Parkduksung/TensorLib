package com.example.tensor

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.widget.Toast
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.jvm.Throws

class Tensor : Application() {

    private val interpreter by lazy {
        Interpreter(
            loadModelFile(),
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

    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor =
            App.instance.context().assets.openFd(Model_IMAGE_SEGMENTATION)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        fileDescriptor.close()
        return retFile
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
        private const val Model_IMAGE_SEGMENTATION = "deeplabv3_257_mv_gpu.tflite"

        const val NUM_CLASSES = 21
        const val IMAGE_SIZE = 257
        const val TO_FLOAT = 4
    }


    fun test(context: Context) {
        Toast.makeText(context, "연결됨..", Toast.LENGTH_SHORT).show()
    }
}