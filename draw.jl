using Luxor

function dump(fig)
    if fig.surfacetype == :png
        Cairo.write_to_png(fig.surface, fig.buffer)
    end

    Cairo.finish(fig.surface)
    Cairo.destroy(fig.surface)

    if fig.filename != ""
        write(fig.filename, fig.bufferdata)
    end
    true
end

function CNOT()
    d = Drawing(400, 400, :png, "cnot")
    origin()

    rect()
    dump(d)
end

@png begin
    fontsize(50)
    circle(Point(0, 0), 150, :stroke)
    text("hello world", halign=:center, valign=:middle)
end
