from typing import List, Dict


def gen_html(nodes: List[Dict], links: List[Dict], file_path: str):
    """nodes为有向图的结点列表，links为有向图的边列表，可视化HTML文件写入file_path
    nodes和links填写参考echarts文档"""
    html = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Awesome-pyecharts</title>
        <script type="text/javascript" src="https://assets.pyecharts.org/assets/echarts.min.js"></script>
    </head>
    <body>
        <div id="main" style="width:1800px; height:800px;"></div>
        <script>
            var myChart = echarts.init(
                document.getElementById('main'), 'white', {renderer: 'canvas'});
            var option = {
                "animation": true,
                "animationThreshold": 2000,
                "animationDuration": 1000,
                "animationEasing": "cubicOut",
                "animationDelay": 0,
                "animationDurationUpdate": 300,
                "animationEasingUpdate": "cubicOut",
                "animationDelayUpdate": 0,
                "color": [
                    "#c23531",
                    "#2f4554",
                    "#61a0a8",
                    "#d48265",
                    "#749f83",
                    "#ca8622",
                    "#bda29a",
                    "#6e7074",
                    "#546570",
                    "#c4ccd3",
                    "#f05b72",
                    "#ef5b9c",
                    "#f47920",
                    "#905a3d",
                    "#fab27b",
                    "#2a5caa",
                    "#444693",
                    "#726930",
                    "#b2d235",
                    "#6d8346",
                    "#ac6767",
                    "#1d953f",
                    "#6950a1",
                    "#918597"
                ],
            "series": [
                {
                    "type": "graph",
                    "layout": "force",
                    "symbol": "rect",  // 把标识点设置为矩形，以便label在上面显示，矩形长度在data中设置
                    "circular": {
                        "rotateLabel": false
                    },
                    "force": {
                        "repulsion": 1000,  // 斥力较大会让点之间较为分散
                        "layoutAnimation": false
                    },
                    "label": {
                        "show": true,
                        "position": "inside",
                        //"backgroundColor": "auto",
                        //"margin": 2,
                        //"padding": 2
                    },
                    "lineStyle": {
                        "width": 1,
                        "opacity": 1,
                        "curveness": 0,
                        "type": "solid"
                    },
                    "roam": true,
                    "focusNodeAdjacency": true,
                    "data": """ + str(nodes) + """,
                    "edgeSymbol": [  // 边的一头是空，另一头是箭头
                        "none",
                        "arrow"
                    ],
                    "draggable": true,  // 支持拖拽
                    "edgeSymbolSize": 10,
                    "links": """ + str(links) + """,
                }],
                "legend": [
                    {
                        "data": [],
                        "selected": {}
                    }
                ],
                "tooltip": {
                    "show": true,
                    "trigger": "item",
                    "triggerOn": "mousemove|click",
                    "axisPointer": {
                        "type": "line"
                    },
                    "textStyle": {
                        "fontSize": 14
                    },
                    "borderWidth": 0
                }
            };
            myChart.setOption(option);
            // 支持拖拽不恢复原位
            myChart.on('mouseup',function(params){
                var option=myChart.getOption();
                option.series[0].data[params.dataIndex].x=params.event.offsetX;
                option.series[0].data[params.dataIndex].y=params.event.offsetY;
                option.series[0].data[params.dataIndex].fixed=true;
                myChart.setOption(option);
            });
        </script>
    </body>
</html>
"""
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    gen_html([{"name": "1"}, {"name": "2"}], [{"source": "1", "target": "2"}])
