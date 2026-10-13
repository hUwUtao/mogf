fn mogf_vertex(input: VertexOut) -> VertexOut {
    var output = input;
    let wave = sin(input.position.x * 14.0 + input.position.w * 0.7) * 0.015;
    output.position.y += wave * input.position.w;
    return output;
}
