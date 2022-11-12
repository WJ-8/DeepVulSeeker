@main def main() = {
import java.io.PrintWriter
val src = cpg.method.name("av_opencl_buffer_write")
  .local
  .referencingIdentifiers
val sink = cpg.call.name("memcpy")
.argument
val aaa = sink.reachableByFlows(src).p
val out = new PrintWriter("result.txt")
out.println(aaa)
out.close()
}
