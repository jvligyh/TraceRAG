// Live OCR UI acceptance. Needs a running service, OCR fixtures, DeepSeek, Playwright and Edge.
const {chromium}=require('playwright');
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');

(async()=>{
  const base='http://127.0.0.1:8000';
  const output=path.resolve(__dirname,'../results/ocr-browser');
  fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const context=await browser.newContext({viewport:{width:1280,height:980},recordVideo:{dir:output,size:{width:1280,height:980}}});
  const page=await context.newPage();
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  const list=await(await context.request.get(base+'/documents')).json();
  const image=list.documents.find(doc=>doc.filename==='OCR中英实验记录.png');
  const pdf=list.documents.find(doc=>doc.filename==='OCR扫描实验记录.pdf');
  assert.ok(image&&pdf,'Expected the actual OCR image and scan PDF fixtures');
  await page.goto(base);
  await page.getByText('本地服务已连接',{exact:false}).waitFor();
  assert.match(await page.locator('#status').textContent(),/OCR已就绪/);
  await page.locator('#all-documents').uncheck();
  await page.getByRole('checkbox',{name:`查询 ${image.filename}`,exact:true}).check();

  async function ask(question,asset){
    await page.locator('#question').fill(question);
    const waiting=page.waitForResponse(response=>response.url().endsWith('/query')&&response.request().method()==='POST',{timeout:240000});
    await page.locator('#ask').click();
    const response=await waiting;
    assert.equal(response.status(),200);
    const data=await response.json();
    assert.equal(data.status,'answered');
    assert.ok(data.citations.length);
    assert.ok(data.results.every(item=>item.asset_id===asset),'Unselected source leaked into OCR answer');
    assert.ok(data.citations.every(item=>item.extraction_method==='ocr'));
    await page.locator('#message').filter({hasText:'查询完成'}).waitFor();
    const label=await page.locator('#citations').textContent();
    assert.match(label,/OCR识别文本，请核对原图/);
    assert.match(label,/不等于正确率/);
    return data;
  }

  const imageAnswer=await ask('图片里记录的光电转换效率是多少？',image.document_id);
  assert.match(imageAnswer.answer,/23[.．]5\s*%|百分之二十三点五/);
  const imageSource=imageAnswer.citations[0].source_url;
  const imageResponse=await context.request.get(base+imageSource.split('#')[0]);
  assert.equal(imageResponse.status(),200);
  assert.equal(imageResponse.headers()['content-type'],'image/png');
  assert.equal((await imageResponse.body()).subarray(0,8).toString('hex'),'89504e470d0a1a0a');
  const [imagePopup]=await Promise.all([context.waitForEvent('page'),page.locator('#citations a').first().click()]);
  await imagePopup.waitForURL(url=>url.href.includes(`/documents/${image.document_id}/file`));
  const imageOpenedUrl=imagePopup.url();
  await imagePopup.close();
  await page.screenshot({path:path.join(output,'image-answer.png'),fullPage:true});

  await page.getByRole('checkbox',{name:`查询 ${image.filename}`,exact:true}).uncheck();
  await page.getByRole('checkbox',{name:`查询 ${pdf.filename}`,exact:true}).check();
  const pdfAnswer=await ask('扫描实验记录中，催化剂的反应温度是多少？',pdf.document_id);
  assert.match(pdfAnswer.answer,/42/);
  const citation=pdfAnswer.citations.find(item=>item.page_number===3);
  assert.ok(citation,'Expected the catalyst record on scan page 3');
  assert.ok(citation.source_url.endsWith('#page=3'));
  const pdfResponse=await context.request.get(base+citation.source_url.split('#')[0]);
  assert.equal(pdfResponse.status(),200);
  assert.equal(pdfResponse.headers()['content-type'],'application/pdf');
  assert.equal((await pdfResponse.body()).subarray(0,5).toString(),'%PDF-');
  const sourceLink=page.locator(`#citations a[href="${citation.source_url}"]`).first();
  const [pdfPopup]=await Promise.all([context.waitForEvent('page'),sourceLink.click()]);
  await pdfPopup.waitForURL(url=>url.href.endsWith('#page=3'));
  const pdfOpenedUrl=pdfPopup.url();
  await pdfPopup.close();
  await page.screenshot({path:path.join(output,'scan-answer.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  const video=page.video();
  await context.close();
  const recording=await video.path();
  await browser.close();
  const report={status:'passed',image_asset:image.document_id,pdf_asset:pdf.document_id,
    image_answer:imageAnswer,pdf_answer:pdfAnswer,image_original_url:imageOpenedUrl,pdf_original_url:pdfOpenedUrl,
    page_errors:errors,mobile_overflow:false,recording};
  fs.writeFileSync(path.join(output,'browser-validation.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify({status:'passed',image_answer:imageAnswer.answer,pdf_answer:pdfAnswer.answer,
    pdf_page:3,image_original_url:imageOpenedUrl,pdf_original_url:pdfOpenedUrl,errors,recording}));
})().catch(error=>{console.error(error.stack);process.exitCode=1});
