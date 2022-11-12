import scala.io.Source
@main def main() = {

val source = Source.fromFile("../train.jsonl","utf-8")
val lineIterator = source.getLines
for(i <- lineIterator){

}

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
