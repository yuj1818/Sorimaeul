package com.usagi.sorimaeul.repository;

import com.querydsl.core.types.Projections;
import com.querydsl.core.types.dsl.BooleanExpression;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.usagi.sorimaeul.dto.dto.ModelInfoDto;
import com.usagi.sorimaeul.entity.QVoiceModel;
import com.usagi.sorimaeul.entity.User;
import com.usagi.sorimaeul.entity.VideoSource;
import com.usagi.sorimaeul.entity.VoiceModel;
import jakarta.persistence.EntityManager;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
@RequiredArgsConstructor
public class VoiceModelRepositoryImpl implements VoiceModelRepositoryCustom {

    private final JPAQueryFactory queryFactory;
    private final QVoiceModel qVoiceModel = QVoiceModel.voiceModel;

    // 유저가 학습 시킨 모델 가져오기(userCode 일치하는 모델)
    public List<ModelInfoDto> userModelList(User user, int state) {
        BooleanExpression stateCondition;
        if (state == 3) {
            stateCondition = qVoiceModel.state.eq(3);
        } else {
            stateCondition = qVoiceModel.state.ne(-1);
        }

        return queryFactory
                .select(Projections.constructor(ModelInfoDto.class, qVoiceModel.modelCode, qVoiceModel.modelName, qVoiceModel.imagePath, qVoiceModel.recordCount, qVoiceModel.state))
                .from(qVoiceModel)
                .where(qVoiceModel.user.eq(user)
                        .and(stateCondition))
                .fetch();
    }
    // 영상 소스 제공 모델 가져오기(videoSourceCode 일치하는 모델)
    public List<ModelInfoDto> videoSourceModelList(VideoSource videoSource) {
        return queryFactory
                .select(Projections.constructor(ModelInfoDto.class, qVoiceModel.modelCode, qVoiceModel.modelName, qVoiceModel.imagePath, qVoiceModel.recordCount, qVoiceModel.state))
                .from(qVoiceModel)
                .where(qVoiceModel.videoSource.eq(videoSource)
                        .and(qVoiceModel.state.eq(3)))
                .fetch();
    }

    public List<ModelInfoDto> commonModelList() {
        return queryFactory
                .select(Projections.constructor(ModelInfoDto.class, qVoiceModel.modelCode, qVoiceModel.modelName, qVoiceModel.imagePath, qVoiceModel.recordCount, qVoiceModel.state))
                .from(qVoiceModel)
                .where(qVoiceModel.user.userCode.isNull()
                        .and(qVoiceModel.videoSource.videoSourceCode.isNull()))
                .fetch();
    }
}
