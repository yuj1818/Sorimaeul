package com.usagi.sorimaeul.api.controller;

import com.usagi.sorimaeul.api.service.ModelService;
import com.usagi.sorimaeul.dto.request.ModelTableCreateRequest;
import com.usagi.sorimaeul.dto.request.ModelUpdateRequest;
import com.usagi.sorimaeul.dto.response.ModelInfoResponse;
import com.usagi.sorimaeul.dto.response.ModelListResponse;
import com.usagi.sorimaeul.dto.response.ModelTableCreateResponse;
import com.usagi.sorimaeul.utils.JwtTokenProvider;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/model")
@RequiredArgsConstructor
@Tag(name = "Model 컨트롤러", description = "음성 모델 관리를 위한 API")
public class ModelController {

    private final ModelService modelService;
    private final JwtTokenProvider jwtTokenProvider;

    @Operation(summary = "음성 모델 테이블 생성", description = "음성 모델 테이블 생성")
    @ApiResponse(responseCode = "201", description = "음성 모델 테이블 생성 성공")
    @PostMapping
    public ResponseEntity<ModelTableCreateResponse> createModelTable(@RequestHeader("Authorization") String token,
                                                                     @RequestBody ModelTableCreateRequest request) {
        long userCode = Long.parseLong(jwtTokenProvider.getPayload(token.substring(7)));
        return modelService.createModelTable(request, userCode);
    }


    @Operation(summary = "음성 녹음", description = "음성 녹음 파일 업로드")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "업로드 성공"),
            @ApiResponse(responseCode = "400", description = "실패")
    })
    @PostMapping("/record/{modelCode}/{num}")
    public ResponseEntity<String> uploadRecordFile(@RequestHeader("Authorization") String token,
                                             @PathVariable int modelCode, @PathVariable int num, MultipartFile recordingFile) {
        long userCode = Long.parseLong(jwtTokenProvider.getPayload(token.substring(7)));
        return modelService.uploadRecordFile(modelCode, num, userCode, recordingFile);
    }


    @Operation(summary = "외부 음성 업로드", description = "외부 음성 녹음 파일을 여러개 업로드 한다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "업로드 성공"),
            @ApiResponse(responseCode = "400", description = "실패")
    })
    @PostMapping("/voice/{modelCode}")
    public ResponseEntity<String> uploadExRecordFile(@RequestHeader("Authorization") String token,
                                                     @PathVariable int modelCode, MultipartFile[] files) {
        long userCode = Long.parseLong(jwtTokenProvider.getPayload(token.substring(7)));
        return modelService.uploadExRecordFile(modelCode, userCode, files);
    }


    @Operation(summary = "외부 모델 업로드", description = "외부 음성 모델 파일을 업로드 한다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "업로드 성공"),
            @ApiResponse(responseCode = "400", description = "실패")
    })
    @PostMapping("/external/{modelCode}")
    public ResponseEntity<String> uploadModelFile(@RequestHeader("Authorization") String token,
                                                     @PathVariable int modelCode, MultipartFile[] modelFiles) {
        long userCode = Long.parseLong(jwtTokenProvider.getPayload(token.substring(7)));
        return modelService.uploadModelFile(modelCode, userCode, modelFiles);
    }


    @Operation(summary = "음성 모델 학습", description = "음성 모델을 학습한다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "학습 성공"),
            @ApiResponse(responseCode = "400", description = "실패")
    })
    @PostMapping("/start/{modelCode}")
    public ResponseEntity<String> learnVoiceModel(@RequestHeader("Authorization") String token,
                                                  @PathVariable int modelCode) {
        long userCode = Long.parseLong(jwtTokenProvider.getPayload(token.substring(7)));
        return modelService.learnVoiceModel(modelCode, userCode);
    }


    @Operation(summary = "모델 리스트 조회", description = "모델 리스트를 조회한다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "모델 리스트 조회 성공"),
            @ApiResponse(responseCode = "400", description = "잘못된 요청입니다.")
    })
    @GetMapping
    public ResponseEntity<ModelListResponse> getModelResponse(@RequestHeader("Authorization") String token,
                                                              @RequestParam(required = false) Integer page, @RequestParam(required = false) Integer videoSourceCode) {
        long userCode = Long.parseLong(jwtTokenProvider.getPayload(token.substring(7)));
        return modelService.getModelList(page, userCode, videoSourceCode);
    }


    @Operation(summary = "모델 상세 조회", description = "모델 상세 조회")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "모델 상세 조회 성공"),
            @ApiResponse(responseCode = "400", description = "잘못된 요청입니다.")
    })
    @GetMapping("/detail/{modelCode}")
    public ResponseEntity<ModelInfoResponse> getModelInfo(@RequestHeader("Authorization") String token,
                                                          @PathVariable int modelCode) {
        long userCode = Long.parseLong(jwtTokenProvider.getPayload(token.substring(7)));
        return modelService.getModelInfo(modelCode, userCode);
    }


    @Operation(summary = "모델 수정", description = "모델명 혹은 모델 대표 이미지 수정")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "모델 수정 성공"),
            @ApiResponse(responseCode = "400", description = "잘못된 요청입니다.")
    })
    @PatchMapping("/detail/{modelCode}")
    public HttpStatus updateModel(@RequestHeader("Authorization") String token,
                                  @PathVariable int modelCode,
                                  @RequestBody ModelUpdateRequest request) {
        long userCode = Long.parseLong(jwtTokenProvider.getPayload(token.substring(7)));
        return modelService.updateModel(modelCode, userCode, request);
    }
}
