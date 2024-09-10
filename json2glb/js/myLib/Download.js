//在浏览器，下载3D模型数据、2D贴图数据
function Download(){//无后缀名的时候，图片会自动填补出正确的后缀名，但模型填补出的是txt后缀名
}
Download.prototype={
    //3D模型数据
    meshDownload:function (mesh,name){
        if(name.charAt(name.length-1)!=='f')name+='.gltf';
        var scene=new THREE.Scene();
        scene.add(mesh);
        this.sceneDownload(scene,name);
    },
    sceneDownload:function (scene,name){
        if(name.charAt(name.length-1)!=='f')name+='.gltf';
        var scope=this;
        var gltfExporter = new THREE.GLTFExporter();
        gltfExporter.parse(scene, function (result) {
            console.log(result)
            scope.jsonDownload(result,name);
        });
    },
    //2D贴图数据
    canvasDownload:function(canvas,name){
        var isPNG=true;
        let url;
        if(isPNG)url = canvas.toDataURL("image/png");//得到图片的base64编码数据
        else url = canvas.toDataURL("image/jpeg");

        //a标签定义超链接，用于从一张页面链接到另一张页面
        let a = document.createElement("a"); // 生成一个a元素
        a.download = name;//name || "photo"; // 设置图片名称
        a.href = url; // 将生成的URL设置为a.href属性
        let event = new MouseEvent("click"); // 创建一个单击事件
        a.dispatchEvent(event); // 触发a的单击事件

        return url.length;
    },
    canvasToUrl:function(canvas){
        var isPNG=true;
        let url;
        if(isPNG)url = canvas.toDataURL("image/png");//得到图片的base64编码数据
        else url = canvas.toDataURL("image/jpeg");
        return url;
    },
    urlDownload:function(url,name){
        //a标签定义超链接，用于从一张页面链接到另一张页面
        let a = document.createElement("a"); // 生成一个a元素
        a.download = name;//name || "photo"; // 设置图片名称
        a.href = url; // 将生成的URL设置为a.href属性
        let event = new MouseEvent("click"); // 创建一个单击事件
        a.dispatchEvent(event); // 触发a的单击事件
    },
    //1D文本数据
    jsonDownload:function (json,name) {
        return this.strDownload(
            JSON.stringify(json),
            name
        );
    },
    strDownload:function (str,name) {//无后缀名
        var myBlob=new Blob([str], { type: 'text/plain' })
        let link = document.createElement('a');
        link.href = URL.createObjectURL(myBlob);
        link.download = name;
        link.click();
        return myBlob.size;
    },
}
